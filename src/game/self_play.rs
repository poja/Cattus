use itertools::Itertools;

use crate::game::common::{GameColor, GamePosition, IGame};
use crate::game::mcts::MCTSPlayer;
use core::panic;
use std::fs;
use std::path;
use std::sync::Arc;
use std::thread;

pub trait PlayerBuilder<Game: IGame>: Sync + Send {
    fn new_player(&self) -> MCTSPlayer<Game>;
}

pub trait DataSerializer<Game: IGame>: Sync + Send {
    fn serialize_data_entry_to_file(
        &self,
        pos: Game::Position,
        probs: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
        filename: String,
    ) -> std::io::Result<()>;
}

pub struct SelfPlayRunner<Game: IGame> {
    player_builder: Arc<dyn PlayerBuilder<Game>>,
    serializer: Arc<dyn DataSerializer<Game>>,
    thread_num: u32,
}

impl<Game: IGame + 'static> SelfPlayRunner<Game> {
    pub fn new(
        player_builder: Box<dyn PlayerBuilder<Game>>,
        serializer: Box<dyn DataSerializer<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player_builder: Arc::from(player_builder),
            serializer: Arc::from(serializer),
            thread_num: thread_num,
        }
    }

    pub fn generate_data(&self, games_num: u32, output_dir: &String) -> std::io::Result<()> {
        /* Create output dir if doesn't exists */
        if !path::Path::new(output_dir).is_dir() {
            fs::create_dir_all(output_dir)?;
        }
        let is_empty = path::PathBuf::from(output_dir).read_dir()?.next().is_none();
        if !is_empty {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "output dir is not empty",
            ));
        }

        let job_builder = |thread_idx| {
            let start_idx = games_num * thread_idx / self.thread_num;
            let end_idx = games_num * (thread_idx + 1) / self.thread_num;

            let worker = SelfPlayWorker::new(
                Arc::clone(&self.player_builder),
                Arc::clone(&self.serializer),
                output_dir.to_string(),
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

        return Ok(());
    }
}

struct SelfPlayWorker<Game: IGame> {
    player_builder: Arc<dyn PlayerBuilder<Game>>,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir: String,
    start_idx: u32,
    end_idx: u32,
}

impl<Game: IGame> SelfPlayWorker<Game> {
    fn new(
        player_builder: Arc<dyn PlayerBuilder<Game>>,
        serializer: Arc<dyn DataSerializer<Game>>,
        output_dir: String,
        start_idx: u32,
        end_idx: u32,
    ) -> Self {
        Self {
            player_builder: player_builder,
            serializer: serializer,
            output_dir: output_dir,
            start_idx: start_idx,
            end_idx: end_idx,
        }
    }

    fn generate_data(&self) -> std::io::Result<()> {
        let mut player = self.player_builder.new_player();
        let mut pos_idx = 0;
        for game_idx in self.start_idx..self.end_idx {
            // if games_num < 10 || game_idx % (games_num / 10) == 0 {
            //     let percentage = (((game_idx as f32) / games_num as f32) * 100.0) as u32;
            //     println!("self play {}%", percentage);
            // }
            let mut pos = Game::Position::new();
            let mut pos_probs_pairs: Vec<(Game::Position, Vec<(Game::Move, f32)>)> = Vec::new();

            while !pos.is_over() {
                /* Generate probabilities from MCTS player */
                let moves = player.calc_moves_probabilities(&pos);
                player.clear();
                let m = player.choose_move_from_probabilities(&moves);

                /* Store probabilities */
                pos_probs_pairs.push((pos.clone(), moves));

                /* Advance game position */
                pos = match m {
                    None => {
                        eprintln!("player failed to choose a move");
                        break;
                    }
                    Some(next_move) => pos.get_moved_position(next_move),
                }
            }
            let winner = pos.get_winner();

            for (pos, probs) in pos_probs_pairs {
                self.serializer.serialize_data_entry_to_file(
                    pos,
                    probs,
                    winner,
                    format!(
                        "{}/d{:#08x}_{:#04x}.json",
                        self.output_dir, game_idx, pos_idx
                    ),
                )?;
                pos_idx += 1;
            }
        }
        return Ok(());
    }
}
