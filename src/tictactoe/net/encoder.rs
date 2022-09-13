use tensorflow::Tensor;

use crate::tictactoe::tictactoe_game::{Bitboard, TicTacToePosition, BOARD_SIZE};

/* Encode the position before feeding it to the network */
pub struct Encoder {}

impl Encoder {
    pub fn new() -> Self {
        Self {}
    }
    pub fn encode_position(&self, pos: &TicTacToePosition) -> Tensor<f32> {
        let planes_num = 2;
        let board_size = BOARD_SIZE as u64;

        let mut encoded_position =
            Vec::with_capacity((planes_num * board_size * board_size) as usize);
        let mut encode_plane = |plane: Bitboard| {
            for idx in 0..(BOARD_SIZE * BOARD_SIZE) {
                encoded_position.push(match plane.get(idx) {
                    true => 1.0,
                    false => 0.0,
                });
            }
        };
        encode_plane(pos.pieces_x());
        encode_plane(pos.pieces_y());

        return Tensor::new(&[1, planes_num, board_size, board_size])
            .with_values(&encoded_position)
            .expect("Can't create input tensor");
    }
}
