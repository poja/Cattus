use crate::game::common::GameColor;
use crate::hex::hex_game::{HexPosition, BOARD_SIZE};

/* Encode the position before feeding it to the network */
pub struct Encoder {}

impl Encoder {
    pub fn new() -> Self {
        Self {}
    }
    pub fn encode_position(&self, position: &HexPosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                vec.push(GameColor::to_idx(position.get_tile(r, c)) as f32);
            }
        }
        return vec;
    }
}
