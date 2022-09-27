use crate::game::common::{GameColor, GameMove, GamePosition, IGame};
use crate::game::mcts::MCTSPlayer;
use crate::utils::Builder;
use itertools::Itertools;
use std::fs;
use std::path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

pub trait DataSerializer<Game: IGame>: Sync + Send {
    fn serialize_data_entry(
        &self,
        pos: Game::Position,
        probs: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
        filename: &String,
    ) -> std::io::Result<()>;
}

pub struct SerializerBase {}

impl SerializerBase {
    pub fn write_entry<Game: IGame, const MOVES_NUM: usize>(
        planes: Vec<u64>,
        probs: Vec<(Game::Move, f32)>,
        winner: f32,
        filename: &String,
    ) -> std::io::Result<()> {
        /* Use -1 for illegal moves */
        let mut probs_vec = vec![-1.0f32; MOVES_NUM];

        /* Fill legal moves probabilities */
        for (m, prob) in probs {
            probs_vec[m.to_nn_idx()] = prob;
        }

        /* Write to file */
        return fs::write(
            filename,
            json::object! {
                planes: planes,
                probs: probs_vec,
                winner: winner
            }
            .dump(),
        );
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
    serializer: Arc<dyn DataSerializer<Game>>,
    thread_num: u32,
}

impl<Game: IGame + 'static> SelfPlayRunner<Game> {
    pub fn new(
        player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        serializer: Arc<dyn DataSerializer<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_builder: player1_builder,
            player2_builder: player2_builder,
            serializer: serializer,
            thread_num: thread_num,
        }
    }

    pub fn generate_data(
        &self,
        games_num: u32,
        output_dir1: &String,
        output_dir2: &String,
        data_entries_prefix: &String,
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
                Arc::clone(&self.serializer),
                output_dir1.to_string(),
                output_dir2.to_string(),
                data_entries_prefix.to_string(),
                Arc::clone(&result),
                start_idx,
                end_idx,
            );

            return move || match worker.generate_data() {
                Ok(_) => {}
                Err(e) => panic!("{:?}", e),
            };
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

        return Ok(());
    }
}

struct SelfPlayWorker<Game: IGame> {
    player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir1: String,
    output_dir2: String,
    data_entries_prefix: String,
    results: Arc<GamesResults>,
    start_idx: u32,
    end_idx: u32,
}

impl<Game: IGame> SelfPlayWorker<Game> {
    fn new(
        player1_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        player2_builder: Arc<dyn Builder<MCTSPlayer<Game>>>,
        serializer: Arc<dyn DataSerializer<Game>>,
        output_dir1: String,
        output_dir2: String,
        data_entries_prefix: String,
        results: Arc<GamesResults>,
        start_idx: u32,
        end_idx: u32,
    ) -> Self {
        Self {
            player1_builder: player1_builder,
            player2_builder: player2_builder,
            serializer: serializer,
            output_dir1: output_dir1,
            output_dir2: output_dir2,
            data_entries_prefix: data_entries_prefix,
            results: results,
            start_idx: start_idx,
            end_idx: end_idx,
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
            let mut pos_probs_pairs: Vec<(Game::Position, Vec<(Game::Move, f32)>)> = Vec::new();

            while !game.is_over() {
                let player = &mut match game.get_position().get_turn() {
                    GameColor::Player1 => [&mut player1, &mut player2],
                    GameColor::Player2 => [&mut player2, &mut player1],
                }[(game_idx % 2) as usize];

                /* Generate probabilities from MCTS player */
                let moves = player.calc_moves_probabilities(game.get_position());
                let next_move = player.choose_move_from_probabilities(&moves).unwrap();

                /* Store probabilities */
                pos_probs_pairs.push((game.get_position().clone(), moves));

                /* Advance game position */
                game.play_single_turn(next_move);
            }
            let winner = game.get_winner();

            let mut pos_idx = 0;
            for (pos, probs) in pos_probs_pairs {
                let output_dir = &match pos.get_turn() {
                    GameColor::Player1 => [&self.output_dir1, &self.output_dir2],
                    GameColor::Player2 => [&self.output_dir2, &self.output_dir1],
                }[(game_idx % 2) as usize];

                self.serializer.serialize_data_entry(
                    pos,
                    probs,
                    winner,
                    &format!(
                        "{}/{}{:#08}_{:#03}.json",
                        output_dir, self.data_entries_prefix, game_idx, pos_idx
                    ),
                )?;
                pos_idx += 1;
            }

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
        return Ok(());
    }
}
