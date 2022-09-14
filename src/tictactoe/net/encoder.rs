use crate::game::common::GameColor;
use crate::game::encoder::Encoder;
use crate::tictactoe::tictactoe_game::{
    TicTacToeGame, TicTacToeMove, TicTacToePosition, BOARD_SIZE,
};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Encoder<TicTacToeGame> for SimpleEncoder {
    fn encode_position(&self, position: &TicTacToePosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                vec.push(GameColor::to_idx(position.get_tile(r, c)) as f32);
            }
        }
        return vec;
    }

    fn encode_per_move_probs(&self, moves: &Vec<(TicTacToeMove, f32)>) -> Vec<f32> {
        let mut vec = vec![0.0; (BOARD_SIZE * BOARD_SIZE) as usize];
        for (m, prob) in moves {
            vec[m.to_idx() as usize] = *prob;
        }
        return vec;
    }
}
