use tensorflow::Tensor;

use crate::hex::hex_game::{Bitboard, HexPosition, BOARD_SIZE};

/* Encode the position before feeding it to the network */
pub struct Encoder {}

impl Encoder {
    pub fn new() -> Self {
        Self {}
    }
    pub fn encode_position(&self, position: &HexPosition) -> Tensor<f32> {
        let cpu = true;

        let planes_num = 2;
        let board_size = BOARD_SIZE as u64;

        let mut encoded_position = vec![0.0; (planes_num * board_size * board_size) as usize];
        let mut encode_plane = |plane: Bitboard, plane_idx: u64| {
            for square in 0..(BOARD_SIZE * BOARD_SIZE) {
                let idx = if cpu {
                    square as u64 * planes_num + plane_idx
                } else {
                    plane_idx * board_size * board_size + square as u64
                };
                encoded_position[idx as usize] = match plane.get(square) {
                    true => 1.0,
                    false => 0.0,
                };
            }
        };

        encode_plane(position.pieces_red(), 0);
        encode_plane(position.pieces_blue(), 1);

        let dims = if cpu {
            [1, board_size, board_size, planes_num]
        } else {
            [1, planes_num, board_size, board_size]
        };
        return Tensor::new(&dims)
            .with_values(&encoded_position)
            .expect("Can't create input tensor");
    }
}
