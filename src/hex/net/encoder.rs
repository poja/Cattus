use tensorflow::Tensor;

use crate::hex::hex_game::{HexPosition, BOARD_SIZE};
use crate::hex::net::common::{self, PLANES_NUM};

/* Encode the position before feeding it to the network */
pub struct Encoder {}

impl Encoder {
    pub fn new() -> Self {
        Self {}
    }
    pub fn encode_position(&self, pos: &HexPosition) -> Tensor<f32> {
        let cpu = true;
        let board_size = BOARD_SIZE as usize;

        let mut encoded_position = vec![0.0; (PLANES_NUM * board_size * board_size) as usize];
        for (plane_idx, plane) in common::position_to_planes(&pos).into_iter().enumerate() {
            for square in 0..(board_size * board_size) {
                let idx = if cpu {
                    square * PLANES_NUM + plane_idx
                } else {
                    plane_idx * board_size * board_size + square
                };
                encoded_position[idx] = match plane.get(square as u8) {
                    true => 1.0,
                    false => 0.0,
                };
            }
        }

        let dims = if cpu {
            [1, board_size as u64, board_size as u64, PLANES_NUM as u64]
        } else {
            [1, PLANES_NUM as u64, board_size as u64, board_size as u64]
        };
        return Tensor::new(&dims)
            .with_values(&encoded_position)
            .expect("Can't create input tensor");
    }
}
