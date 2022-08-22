use crate::game::common::GameColor;
use crate::game::encoder::Encoder;
use crate::tictactoe::tictactoe_game::{self, TicTacToeGame, TicTacToeMove, TicTacToePosition};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Encoder<TicTacToeGame> for SimpleEncoder {
    fn encode_position(&self, position: &TicTacToePosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..tictactoe_game::BOARD_SIZE {
            for c in 0..tictactoe_game::BOARD_SIZE {
                vec.push(match position.get_tile(r, c) {
                    Some(color) => match color {
                        GameColor::Player1 => 1.0,
                        GameColor::Player2 => -1.0,
                    },
                    None => 0.0,
                });
            }
        }
        return vec;
    }

    fn encode_per_move_probs(&self, moves: &Vec<(TicTacToeMove, f32)>) -> Vec<f32> {
        let mut vec = vec![0.0; tictactoe_game::BOARD_SIZE * tictactoe_game::BOARD_SIZE];
        for (m, prob) in moves {
            let (r, c) = m.cell;
            vec[r * tictactoe_game::BOARD_SIZE + c] = *prob;
        }
        return vec;
    }
}
