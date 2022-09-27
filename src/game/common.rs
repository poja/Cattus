use std::{
    fmt::{Debug, Display},
    hash::Hash,
};

use rand::Rng;

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
}

pub trait IGame {
    type Position: GamePosition<Game = Self>;
    type Move: GameMove<Game = Self>;

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
    fn get_repetition_limit() -> Option<u32>;
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

pub trait Bitboard {
    fn new() -> Self;
    fn new_with_all(val: bool) -> Self;
    fn get(&self, idx: usize) -> bool;
    fn set(&mut self, idx: usize, val: bool);
}

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
