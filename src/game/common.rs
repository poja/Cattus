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

    fn new() -> Self;
    fn new_from_pos(pos: Self::Position) -> Self;
    fn get_position(&self) -> Self::Position;
    fn play_single_turn(&mut self, player: &mut dyn GamePlayer<Self>);
    fn play_until_over(
        &mut self,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>);
}

pub trait GamePosition: Clone + Copy + Eq {
    type Game: IGame<Position = Self>;
    fn new() -> Self;
    fn get_turn(&self) -> GameColor;
    // TODO change get_legal_moves to return iterator
    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move>;
    fn get_moved_position(&self, m: <Self::Game as IGame>::Move)
        -> <Self::Game as IGame>::Position;
    fn is_over(&self) -> bool;
    fn get_winner(&self) -> Option<GameColor>;
}

pub trait GameMove: Clone + Copy + Eq + std::cmp::Eq + std::hash::Hash + std::fmt::Debug {
    type Game: IGame<Move = Self>;
}

pub trait GamePlayer<Game: IGame> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move>;
}
