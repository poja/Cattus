
// TODO this code can be shared for all games

use crate::game::common::{GamePlayer, GamePosition, IGame};
use rand::Rng;

pub struct PlayerRand {}

impl PlayerRand {
    pub fn new() -> Self {
        Self {}
    }
}

impl<Game: IGame> GamePlayer<Game> for PlayerRand {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = position.get_legal_moves();
        if moves.len() == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        return Some(moves[rng.gen_range(0..moves.len()) as usize]);
    }
}
