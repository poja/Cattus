use itertools::Itertools;
use std::fs;
use std::ops::Deref;
use std::path;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
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

impl<Game: IGame> Clone for DataEntry<Game> {
    fn clone(&self) -> Self {
        DataEntry {
            pos: self.pos,
            probs: self.probs.clone(),
            winner: self.winner,
        }
    }
}

pub trait DataSerializer<Game: IGame>: Sync + Send {
    fn serialize_data_entry(&self, entry: DataEntry<Game>, filename: &str) -> std::io::Result<()>;
}

pub struct SerializerBase;
impl SerializerBase {
    pub fn write_entry<Game: IGame>(
        planes: Vec<u64>,
        probs: Vec<(Game::Move, f32)>,
        winner: i8,
        filename: &str,
    ) -> std::io::Result<()> {
        /* Use -1 for illegal moves */
        let mut probs_vec = vec![-1.0f32; Game::MOVES_NUM];

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
        bytes.extend(winner.to_le_bytes());
        assert!(bytes.len() == size);

        /* Write to file */
        fs::write(filename, bytes)
    }
}

#[derive(Copy, Clone)]
pub struct GamesResults {
    pub w1: u32,
    pub w2: u32,
    pub d: u32,
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
    ) -> std::io::Result<GamesResults> {
        assert!(games_num % 2 == 0, "Games num should be a multiple of 2");

        /* Create output dir if doesn't exists */
        for output_dir in [output_dir1, output_dir2] {
            if !path::Path::new(output_dir).is_dir() {
                fs::create_dir_all(output_dir)?;
            }
        }

        let games_counter = Arc::new(AtomicUsize::new(0));
        let result = Arc::new(Mutex::new(GamesResults { w1: 0, w2: 0, d: 0 }));

        let job_builder = || {
            let worker = SelfPlayWorker::new(
                self.player1_builder.clone(),
                self.player2_builder.clone(),
                &self.temperature_policy,
                self.serializer.clone(),
                output_dir1.to_string(),
                output_dir2.to_string(),
                result.clone(),
                games_counter.clone(),
                games_num,
            );

            move || worker.generate_data().unwrap()
        };

        /* Spawn thread_num-1 to jobs [1..thread_num-1] */
        let threads = (1..self.thread_num)
            .map(|_| thread::spawn(job_builder()))
            .collect_vec();

        /* Use current thread to do job 0 */
        job_builder()();

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        let res = *result.lock().unwrap().deref();
        Ok(res)
    }
}

struct SelfPlayWorker<Game: IGame> {
    player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    temperature_scheduler: TemperatureScheduler,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir1: String,
    output_dir2: String,
    results: Arc<Mutex<GamesResults>>,
    games_queue: Arc<AtomicUsize>,
    games_num: usize,
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
        results: Arc<Mutex<GamesResults>>,
        games_queue: Arc<AtomicUsize>,
        games_num: usize,
    ) -> Self {
        Self {
            player1_builder,
            player2_builder,
            temperature_scheduler: TemperatureScheduler::from_str(temperature_policy),
            serializer,
            output_dir1,
            output_dir2,
            results,
            games_queue,
            games_num,
        }
    }

    fn generate_data(&self) -> std::io::Result<()> {
        let mut player1 = self.player1_builder.build();
        let mut player2 = self.player2_builder.build();

        loop {
            let game_idx = self
                .games_queue
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if game_idx >= self.games_num {
                break;
            }

            let mut game = Game::new();
            let mut pos_probs_pairs = Vec::new();
            let players_switch = game_idx % 2 == 1;

            let mut half_move_num = 0;
            while !game.is_over() {
                let mut player = game.get_position().get_turn();
                if players_switch {
                    player = player.opposite()
                }
                let player = match player {
                    GameColor::Player1 => &mut player1,
                    GameColor::Player2 => &mut player2,
                };

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
            {
                let mut results = self.results.lock().unwrap();
                let counter = match winner {
                    None => &mut results.d,
                    Some(mut player) => {
                        if players_switch {
                            player = player.opposite();
                        }
                        match player {
                            GameColor::Player1 => &mut results.w1,
                            GameColor::Player2 => &mut results.w2,
                        }
                    }
                };
                *counter += 1;
            }

            log::debug!("Game {} done", game_idx);
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
        }[game_idx % 2];

        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (probs, winner) = net::flip_score_if_needed((probs, winner), is_flipped);
        let winner = GameColor::from_idx(winner as i32);

        let entries = Game::produce_transformed_data_entries(DataEntry { pos, probs, winner });
        for (transform_idx, entry) in entries.into_iter().enumerate() {
            self.serializer.serialize_data_entry(
                entry,
                &format!(
                    "{}/{:#08}_{:#03}_{:#02}.traindata",
                    output_dir, game_idx, pos_idx, transform_idx
                ),
            )?;
        }
        Ok(())
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
    ///   with a ',' between them.
    ///
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
        for i in 0..((s.len() - 1) / 2) {
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
