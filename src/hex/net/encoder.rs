use crate::game_utils::{game, self_play};
use crate::hex::hex_game::{self, HexGame, HexPosition};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl self_play::Encoder<HexGame> for SimpleEncoder {
    fn encode_moves(&self, _moves: &Vec<(hex_game::Location, f32)>) -> Vec<f32> {
        return vec![];
    }
    fn decode_moves(&self, _moves: &Vec<f32>) -> Vec<(hex_game::Location, f32)> {
        return vec![];
    }
    fn encode_position(&self, position: &HexPosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..hex_game::BOARD_SIZE {
            for c in 0..hex_game::BOARD_SIZE {
                vec.push(match position.get_tile(r, c) {
                    hex_game::Hexagon::Full(color) => match color {
                        game::GameColor::Player1 => 1.0,
                        game::GameColor::Player2 => -1.0,
                    },
                    hex_game::Hexagon::Empty => 0.0,
                });
            }
        }
        return vec;
    }
}
