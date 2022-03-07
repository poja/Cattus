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
}

pub trait IGame {
    type Position: GamePosition<Game = Self>;
    type Move: GameMove<Game = Self>;
    fn play_until_over(
        pos: &Self::Position,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>);
}

pub trait GamePosition: Clone + Copy + Eq {
    type Game: IGame;
    fn get_turn(&self) -> GameColor;
    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move>;
    fn get_moved_position(&self, m: <Self::Game as IGame>::Move)
        -> <Self::Game as IGame>::Position;
    fn is_over(&self) -> bool;
    fn get_winner(&self) -> Option<GameColor>;
}

pub trait GameMove: Clone + Copy + Eq + std::cmp::Eq + std::hash::Hash {
    type Game: IGame;
}

pub trait GamePlayer<Game: IGame> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move>;
}
