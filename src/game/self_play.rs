use itertools::Itertools;
use std::fs;
use std::path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use crate::game::common::{GameColor, GameMove, GamePosition, IGame};
use crate::game::mcts::MCTSPlayer;
use crate::game::net;
use crate::utils::Builder;

pub struct DataEntry<Game: IGame> {
    pub pos: Game::Position,
    pub probs: Vec<(Game::Move, f32)>,
    pub winner: Option<GameColor>,
}

pub trait DataSerializer<Game: IGame>: Sync + Send {
    fn serialize_data_entry(&self, entry: DataEntry<Game>, filename: &str) -> std::io::Result<()>;
}

pub struct SerializerBase;
impl SerializerBase {
    pub fn write_entry<Game: IGame, const MOVES_NUM: usize>(
        planes: Vec<u64>,
        probs: Vec<(Game::Move, f32)>,
        winner: i8,
        filename: &str,
    ) -> std::io::Result<()> {
        /* Use -1 for illegal moves */
        let mut probs_vec = vec![-1.0f32; MOVES_NUM];

        /* Fill legal moves probabilities */
        for (m, prob) in probs {
            probs_vec[m.to_nn_idx()] = prob;
        }

        let u64bytes = u64::BITS as usize / 8;
        let f32bytes = /* f32::BITS */ 32 / 8;
        let i8bytes = i8::BITS as usize / 8;
        let size = planes.len() * u64bytes + probs_vec.len() * f32bytes + i8bytes;
        let mut bytes = Vec::with_capacity(size);

        /* Serialized in little indian format, should deserialized the same */
        bytes.extend(planes.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(probs_vec.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend((winner as i8).to_le_bytes());
        assert!(bytes.len() == size);

        /* Write to file */
        fs::write(filename, bytes)
    }
}

struct GamesResults {
    w1: AtomicU32,
    w2: AtomicU32,
    d: AtomicU32,
}

pub struct SelfPlayRunner<Game: IGame> {
    player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    temperature_policy: String,
    serializer: Arc<dyn DataSerializer<Game>>,
    thread_num: usize,
}

impl<Game: IGame + 'static> SelfPlayRunner<Game> {
    pub fn new(
        player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        temperature_policy: String,
        serializer: Arc<dyn DataSerializer<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_builder,
            player2_builder,
            temperature_policy,
            serializer,
            thread_num: thread_num as usize,
        }
    }

    pub fn generate_data(
        &self,
        games_num: usize,
        output_dir1: &String,
        output_dir2: &String,
        result_file: &String,
    ) -> std::io::Result<()> {
        if games_num % 2 != 0 {
            panic!("Games num should be a multiple of 2");
        }

        /* Create output dir if doesn't exists */
        for output_dir in [output_dir1, output_dir2] {
            if !path::Path::new(output_dir).is_dir() {
                fs::create_dir_all(output_dir)?;
            }
        }

        let result = Arc::new(GamesResults {
            w1: AtomicU32::new(0),
            w2: AtomicU32::new(0),
            d: AtomicU32::new(0),
        });

        let job_builder = |thread_idx| {
            let start_idx = games_num * thread_idx / self.thread_num;
            let end_idx = games_num * (thread_idx + 1) / self.thread_num;

            let worker = SelfPlayWorker::new(
                Arc::clone(&self.player1_builder),
                Arc::clone(&self.player2_builder),
                &self.temperature_policy,
                Arc::clone(&self.serializer),
                output_dir1.to_string(),
                output_dir2.to_string(),
                Arc::clone(&result),
                start_idx,
                end_idx,
            );

            move || match worker.generate_data() {
                Ok(_) => {}
                Err(e) => panic!("{:?}", e),
            }
        };

        /* Spawn thread_num-1 to jobs [1..thread_num-1] */
        let threads = (1..self.thread_num)
            .map(|thread_idx| thread::spawn(job_builder(thread_idx)))
            .collect_vec();

        /* Use current thread to do job 0 */
        job_builder(0)();

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        if result_file != "_NONE_" {
            fs::write(
                result_file,
                json::object! {
                    player1_wins: result.w1.load(Ordering::Relaxed),
                    player2_wins: result.w2.load(Ordering::Relaxed),
                    draws: result.d.load(Ordering::Relaxed),
                }
                .dump(),
            )?;
        }

        Ok(())
    }
}

struct SelfPlayWorker<Game: IGame> {
    player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    temperature_scheduler: TemperatureScheduler,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir1: String,
    output_dir2: String,
    results: Arc<GamesResults>,
    start_idx: usize,
    end_idx: usize,
}

impl<Game: IGame> SelfPlayWorker<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        temperature_policy: &str,
        serializer: Arc<dyn DataSerializer<Game>>,
        output_dir1: String,
        output_dir2: String,
        results: Arc<GamesResults>,
        start_idx: usize,
        end_idx: usize,
    ) -> Self {
        Self {
            player1_builder,
            player2_builder,
            temperature_scheduler: TemperatureScheduler::from_str(temperature_policy),
            serializer,
            output_dir1,
            output_dir2,
            results,
            start_idx,
            end_idx,
        }
    }

    fn generate_data(&self) -> std::io::Result<()> {
        let mut player1 = self.player1_builder.build();
        let mut player2 = self.player2_builder.build();

        for game_idx in self.start_idx..self.end_idx {
            // if games_num < 10 || game_idx % (games_num / 10) == 0 {
            //     let percentage = (((game_idx as f32) / games_num as f32) * 100.0) as u32;
            //     println!("self play {}%", percentage);
            // }
            let mut game = Game::new();
            let mut pos_probs_pairs = Vec::new();

            let mut half_move_num = 0;
            while !game.is_over() {
                let player = &mut match game.get_position().get_turn() {
                    GameColor::Player1 => [&mut player1, &mut player2],
                    GameColor::Player2 => [&mut player2, &mut player1],
                }[(game_idx % 2) as usize];

                /* Generate probabilities from MCTS player */
                player.set_temperature(
                    self.temperature_scheduler
                        .get_temperature((half_move_num / 2) as u32),
                );
                let moves = player.calc_moves_probabilities(game.get_position());
                let next_move = player.choose_move_from_probabilities(&moves).unwrap();

                /* Store probabilities */
                pos_probs_pairs.push((*game.get_position(), moves));

                /* Advance game position */
                game.play_single_turn(next_move);

                half_move_num += 1;
            }

            /* Save all data entries */
            let winner = game.get_winner();
            for (pos_idx, (pos, probs)) in pos_probs_pairs.into_iter().enumerate() {
                self.write_data_entry(game_idx, pos_idx, pos, probs, winner)?;
            }

            /* Update winning counters */
            match winner {
                None => &self.results.d,
                Some(p) => {
                    let res = match p {
                        GameColor::Player1 => [&self.results.w1, &self.results.w2],
                        GameColor::Player2 => [&self.results.w2, &self.results.w1],
                    };
                    res[(game_idx % 2) as usize]
                }
            }
            .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn write_data_entry(
        &self,
        game_idx: usize,
        pos_idx: usize,
        pos: Game::Position,
        probs: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
    ) -> std::io::Result<()> {
        let output_dir = match pos.get_turn() {
            GameColor::Player1 => [&self.output_dir1, &self.output_dir2],
            GameColor::Player2 => [&self.output_dir2, &self.output_dir1],
        }[(game_idx % 2) as usize];

        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (winner, probs) = net::flip_score_if_needed((winner, probs), is_flipped);
        let winner = GameColor::from_idx(winner as i32);

        let entry = DataEntry { pos, probs, winner };
        self.serializer.serialize_data_entry(
            entry,
            &format!("{}/{:#08}_{:#03}.traindata", output_dir, game_idx, pos_idx),
        )
    }
}

struct TemperatureScheduler {
    temperatures: Vec<(u32, f32)>,
    last_temperature: f32,
}

impl TemperatureScheduler {
    /// Create a scheduler from a string describing the temperature policy
    ///
    /// # Arguments
    ///
    /// * `s` - A string representing the temperature policy. The string should contain an odd number of numbers,
    /// with a ',' between them.
    /// The string will be split into pairs of two numbers, and a final number.
    /// Each pair should be of the form (moves_num, temperature) and the final number is the final temperature.
    /// Each pair represent an interval of moves number in which the corresponding temperature will be assigned.
    /// The pairs should be ordered by the moves_num.
    ///
    /// # Examples
    ///
    /// "1.0" means a constant temperature of 1
    /// "30,1.0,0.0" means a temperature of 1.0 for the first 30 moves, and temperature of zero after than
    /// "15,2.0,30,0.5,0.1" means a temperature of 2.0 for the first 15 moves, 0.5 in the moves 16 up to 30, and 0.1
    /// after that
    fn from_str(s: &str) -> Self {
        let s = s.split(',').collect_vec();
        assert!(s.len() % 2 == 1);

        let mut temperatures = Vec::new();
        for i in 0..(((s.len() - 1) / 2) as usize) {
            let threshold = s[i * 2].parse::<u32>().unwrap();
            let temperature = s[i * 2 + 1].parse::<f32>().unwrap();
            if !temperatures.is_empty() {
                let (last_threshold, _last_temp) = temperatures.last().unwrap();
                assert!(*last_threshold < threshold);
            }
            temperatures.push((threshold, temperature));
        }
        let last_temp = s.last().unwrap().parse::<f32>().unwrap();
        Self {
            temperatures,
            last_temperature: last_temp,
        }
    }

    fn get_temperature(&self, move_num: u32) -> f32 {
        for (threshold, temperature) in &self.temperatures {
            if move_num < *threshold {
                return *temperature;
            }
        }
        self.last_temperature
    }
}
