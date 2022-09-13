use crate::game::common::{GameColor, GamePosition, IGame};
use crate::game::encoder::Encoder;
use crate::game::mcts::MCTSPlayer;
use core::panic;
use json;
use std::fs;
use std::path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub trait PlayerBuilder<Game: IGame>: Sync + Send {
    fn new_player(&self) -> MCTSPlayer<Game>;
}

pub struct SelfPlayRunner<Game: IGame> {
    encoder: Arc<dyn Encoder<Game>>,
}

impl<Game: IGame + 'static> SelfPlayRunner<Game> {
    pub fn new(encoder: Box<dyn Encoder<Game>>) -> Self {
        Self {
            encoder: Arc::from(encoder),
        }
    }

    pub fn generate_data(
        &self,
        player_builder: Box<dyn PlayerBuilder<Game>>,
        games_num: u32,
        output_dir: &String,
    ) -> std::io::Result<()> {
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

        let player_builder = Arc::from(player_builder);

        let thread_num = 8;
        let mut threads: Vec<JoinHandle<()>> = vec![];
        for thread_idx in 0..thread_num {
            let start_idx = games_num * thread_idx / thread_num;
            let end_idx = games_num * (thread_idx + 1) / thread_num;

            let worker = SelfPlayWorker::new(
                Arc::clone(&player_builder),
                Arc::clone(&self.encoder),
                output_dir.to_string(),
                start_idx,
                end_idx,
            );
            threads.push(thread::spawn(move || {
                let worker = worker;
                match worker.generate_data() {
                    Ok(_) => {}
                    Err(e) => panic!("{:?}", e),
                }
            }));
        }

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        return Ok(());
    }
}

struct SelfPlayWorker<Game: IGame> {
    player_builder: Arc<dyn PlayerBuilder<Game>>,
    encoder: Arc<dyn Encoder<Game>>,
    output_dir: String,
    start_idx: u32,
    end_idx: u32,
}

impl<Game: IGame> SelfPlayWorker<Game> {
    fn new(
        player_builder: Arc<dyn PlayerBuilder<Game>>,
        encoder: Arc<dyn Encoder<Game>>,
        output_dir: String,
        start_idx: u32,
        end_idx: u32,
    ) -> Self {
        Self {
            player_builder: player_builder,
            encoder: encoder,
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
            let mut pos_move_probs_pairs: Vec<(Game::Position, Vec<(Game::Move, f32)>)> =
                Vec::new();

            while !pos.is_over() {
                /* Generate probabilities from MCTS player */
                let moves = player.calc_moves_probabilities(&pos);
                player.clear();
                let m = player.choose_move_from_probabilities(&moves);

                /* Store probabilities */
                pos_move_probs_pairs.push((pos.clone(), moves));

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

            for (pos, per_move_prob) in pos_move_probs_pairs {
                self.write_data_to_file(
                    pos,
                    per_move_prob,
                    winner,
                    format!("{}/d{:#08x}_{:#04x}.json", self.output_dir, game_idx, pos_idx),
                )?;
                pos_idx += 1;
            }
        }
        return Ok(());
    }

    fn write_data_to_file(
        &self,
        pos: Game::Position,
        per_move_prob: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
        filename: String,
    ) -> std::io::Result<()> {
        let pos_vec = self.encoder.encode_position(&pos);

        let turn = match pos.get_turn() {
            GameColor::Player1 => 1,
            GameColor::Player2 => -1,
        };

        let per_move_prob_vec = self.encoder.encode_per_move_probs(&per_move_prob);

        let winner_int = match winner {
            None => 0,
            Some(winning_player) => match winning_player {
                GameColor::Player1 => 1,
                GameColor::Player2 => -1,
            },
        };

        let json_obj = json::object! {
            position: pos_vec,
            turn: turn,
            moves_probabilities: per_move_prob_vec,
            winner: winner_int
        };

        let json_str = json_obj.dump();
        fs::write(filename, json_str)?;

        return Ok(());
    }
}
