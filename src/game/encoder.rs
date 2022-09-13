use crate::game::common::IGame;

pub trait Encoder<Game: IGame>: Sync + Send {
    fn encode_position(&self, position: &Game::Position) -> Vec<f32>;
    fn encode_per_move_probs(&self, moves: &Vec<(Game::Move, f32)>) -> Vec<f32>;
}
