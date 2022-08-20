use crate::game::common::GameColor;
use crate::game::encoder::Encoder;
use crate::hex::hex_game::{self, HexGame, HexPosition};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Encoder<HexGame> for SimpleEncoder {
    fn encode_position(&self, position: &HexPosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..hex_game::BOARD_SIZE {
            for c in 0..hex_game::BOARD_SIZE {
                vec.push(match position.get_tile(r, c) {
                    hex_game::Hexagon::Full(color) => match color {
                        GameColor::Player1 => 1.0,
                        GameColor::Player2 => -1.0,
                    },
                    hex_game::Hexagon::Empty => 0.0,
                });
            }
        }
        return vec;
    }
    fn encode_per_move_probs(&self, moves: &Vec<(hex_game::Location, f32)>) -> Vec<f32> {
        let mut vec = vec![0.0; hex_game::BOARD_SIZE * hex_game::BOARD_SIZE];
        for ((r, c), prob) in moves {
            vec[r * hex_game::BOARD_SIZE + c] = *prob;
        }
        return vec;
    }
}
