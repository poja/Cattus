use rand::prelude::*;

use std::fmt::{Debug, Display};
use std::hash::Hash;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GameColor {
    Player1,
    Player2,
}

impl GameColor {
    pub fn opposite(&self) -> GameColor {
        match self {
            GameColor::Player1 => GameColor::Player2,
            GameColor::Player2 => GameColor::Player1,
        }
    }

    pub fn to_idx(player: Option<GameColor>) -> i32 {
        match player {
            Some(GameColor::Player1) => 1,
            Some(GameColor::Player2) => -1,
            None => 0,
        }
    }

    pub fn from_idx(player: i32) -> Option<GameColor> {
        match player {
            1 => Some(GameColor::Player1),
            -1 => Some(GameColor::Player2),
            0 => None,
            other => panic!("unknown player index: {}", other),
        }
    }
}

pub trait IGame: Sized {
    type Position: GamePosition<Game = Self>;
    type Move: GameMove<Game = Self>;
    type Bitboard: GameBitboard<Game = Self>;
    const BOARD_SIZE: usize;
    const MOVES_NUM: usize;
    const REPETITION_LIMIT: Option<usize>;

    fn new() -> Self;
    fn new_from_pos(pos: Self::Position) -> Self;
    fn get_position(&self) -> &Self::Position;
    fn is_over(&self) -> bool;
    fn get_winner(&self) -> Option<GameColor>;
    fn play_single_turn(&mut self, next_move: Self::Move);
    fn play_until_over(
        &mut self,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>);
}

pub trait GamePosition: Clone + Copy + Eq + Hash + Send + Sync {
    type Game: IGame<Position = Self>;

    fn new() -> Self;
    fn get_turn(&self) -> GameColor;
    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move>;
    fn get_moved_position(&self, m: <Self::Game as IGame>::Move) -> Self;
    fn is_over(&self) -> bool;
    fn get_winner(&self) -> Option<GameColor>;
    fn get_flip(&self) -> Self;
    fn print(&self);
}

pub trait GameMove: Clone + Copy + Eq + Hash + Display + Debug + Send + Sync {
    type Game: IGame<Move = Self>;

    fn get_flip(&self) -> Self;
    fn to_nn_idx(&self) -> usize;
}

pub trait GamePlayer<Game: IGame> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move>;
}

pub trait GameBitboard: Clone + Copy {
    type Game: IGame<Bitboard = Self>;

    fn new() -> Self;
    fn new_with_all(val: bool) -> Self;
    fn get(&self, idx: usize) -> bool;
    fn set(&mut self, idx: usize, val: bool);
}

pub struct PlayerRand {
    rand: StdRng,
}
impl Default for PlayerRand {
    fn default() -> Self {
        Self::new()
    }
}
impl PlayerRand {
    pub fn new() -> Self {
        Self::from_seed(rand::rng().random())
    }

    pub fn from_seed(seed: u64) -> Self {
        Self {
            rand: StdRng::seed_from_u64(seed),
        }
    }
}

impl<Game: IGame> GamePlayer<Game> for PlayerRand {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = position.get_legal_moves();
        if moves.is_empty() {
            None
        } else {
            Some(moves[self.rand.random_range(0..moves.len())])
        }
    }
}
