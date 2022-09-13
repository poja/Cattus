use crate::game::common::{GameColor, GamePosition, IGame};
use crate::game::encoder::Encoder;
use crate::game::mcts::MCTSPlayer;
use json;
use std::fs;
use std::path;

pub trait PlayerBuilder<Game: IGame>: Sync + Send {
    fn new_player(&self) -> MCTSPlayer<Game>;
}

pub struct SelfPlayRunner<Game: IGame> {
    encoder: Box<dyn Encoder<Game>>,
}

impl<Game: IGame> SelfPlayRunner<Game> {
    pub fn new(encoder: Box<dyn Encoder<Game>>) -> Self {
        Self { encoder: encoder }
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
        let mut data_idx: u64 = 0;

        let mut player = player_builder.new_player();

        for game_idx in 0..games_num {
            if games_num < 10 || game_idx % (games_num / 10) == 0 {
                let percentage = (((game_idx as f32) / games_num as f32) * 100.0) as u32;
                println!("self play {}%", percentage);
            }
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
                    format!("{}/d{:#016x}.json", output_dir, data_idx),
                )?;
                data_idx += 1;
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
