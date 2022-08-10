use crate::game_utils::game::{GameColor, GamePosition, IGame};
use crate::game_utils::mcts::MCTSPlayer;
use json;
use std::fs;
use std::path;

pub trait Encoder<Game: IGame> {
    fn encode_moves(&self, moves: &Vec<(Game::Move, f32)>) -> Vec<f32>;
    fn decode_moves(&self, moves: &Vec<f32>) -> Vec<(Game::Move, f32)>;
    fn encode_position(&self, position: &Game::Position) -> Vec<f32>;
}

pub struct TrainData {
    pos: Vec<f32>,
    turn: i8,
    moves_probabilities: Vec<f32>,
    winner: i8,
}

impl TrainData {
    pub fn new(pos: Vec<f32>, turn: i8, moves_probabilities: Vec<f32>, winner: i8) -> Self {
        Self {
            pos: pos,
            turn: turn,
            moves_probabilities: moves_probabilities,
            winner: winner,
        }
    }
}

pub struct SelfPlayRunner<'a, Game: IGame> {
    encoder: &'a dyn Encoder<Game>,
}

impl<'a, Game: IGame> SelfPlayRunner<'a, Game> {
    pub fn new(encoder: &'a dyn Encoder<Game>) -> Self {
        Self { encoder: encoder }
    }

    pub fn generate_data(
        &self,
        player: &mut MCTSPlayer<Game>,
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

        for _ in 0..games_num {
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
                println!("advancing another step in game");
                pos = match m {
                    None => {
                        eprintln!("player failed to choose a move");
                        break;
                    }
                    Some(next_move) => pos.get_moved_position(next_move),
                }
            }
            let winner = pos.get_winner();

            for pos_move_probs_pair in pos_move_probs_pairs {
                let data = TrainData::new(
                    self.encoder.encode_position(&pos_move_probs_pair.0),
                    match pos_move_probs_pair.0.get_turn() {
                        GameColor::Player1 => 1,
                        GameColor::Player2 => -1,
                    },
                    self.encoder.encode_moves(&pos_move_probs_pair.1),
                    match winner {
                        None => 0,
                        Some(winning_player) => match winning_player {
                            GameColor::Player1 => 1,
                            GameColor::Player2 => -1,
                        },
                    },
                );

                let filename = format!("{}/d{:#016x}.json", output_dir, data_idx);
                println!("Writing game pos to dick: {}", filename);
                self.write_data_to_file(data, filename)?;
                data_idx += 1;
            }
        }

        return Ok(());
    }

    fn write_data_to_file(&self, data: TrainData, filename: String) -> std::io::Result<()> {
        let json_obj = json::object! {
            position: data.pos,
            turn: data.turn,
            moves_probabilities: data.moves_probabilities,
            winner: data.winner
        };
        let json_str = json_obj.dump();
        fs::write(filename, json_str)?;
        return Ok(());
    }
}
