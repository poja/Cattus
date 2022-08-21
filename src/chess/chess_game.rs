use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame};
use chess;
use itertools::Itertools;

pub const BOARD_SIZE: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChessMove {
    m: chess::ChessMove,
}
impl ChessMove {
    pub fn new(m: chess::ChessMove) -> Self {
        Self { m: m }
    }
}
impl GameMove for ChessMove {
    type Game = ChessGame;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChessPosition {
    board: chess::Board,
}

impl ChessPosition {
    pub fn new() -> Self {
        Self {
            board: chess::Board::default(),
        }
    }
    fn new_from_board(board: chess::Board) -> Self {
        Self { board: board }
    }

    // pub fn flip_of(pos: &ChessPosition) -> Self {}

    pub fn is_valid_move(&self, m: ChessMove) -> bool {
        self.board.legal(m.m)
    }

    // pub fn get_tile(&self, r: usize, c: usize) -> Hexagon {
    //     assert!(r < BOARD_SIZE && c < BOARD_SIZE);
    //     self.board[r][c]
    // }

    // pub fn print(&self) -> () {
    //     for row_i in 0..BOARD_SIZE {
    //         let row_characters: Vec<String> = self.board[row_i]
    //             .iter()
    //             .map(|hex| String::from(hex.char()))
    //             .collect();
    //         let spaces = " ".repeat(BOARD_SIZE - row_i - 1);
    //         println!("{}{}", spaces, row_characters.join(" "));
    //     }
    // }
}

fn chess_color_to_game_color(c: chess::Color) -> GameColor {
    match c {
        chess::Color::White => GameColor::Player1,
        chess::Color::Black => GameColor::Player2,
    }
}

impl GamePosition for ChessPosition {
    type Game = ChessGame;
    fn new() -> Self {
        ChessPosition::new()
    }
    fn get_turn(&self) -> GameColor {
        chess_color_to_game_color(self.board.side_to_move())
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        chess::MoveGen::new_legal(&self.board)
            .map(|m| ChessMove::new(m))
            .collect_vec()
    }

    fn get_moved_position(
        &self,
        m: <Self::Game as IGame>::Move,
    ) -> <Self::Game as IGame>::Position {
        ChessPosition::new_from_board(self.board.make_move_new(m.m))
    }

    fn is_over(&self) -> bool {
        self.board.status() != chess::BoardStatus::Ongoing
    }

    fn get_winner(&self) -> Option<GameColor> {
        return match self.board.status() {
            chess::BoardStatus::Ongoing => panic!("Game is not over"),
            chess::BoardStatus::Stalemate => None,
            chess::BoardStatus::Checkmate => {
                // TODO not sure this is correct, need to check.
                // looks valid according to https://docs.rs/chess/latest/src/chess/game.rs.html#98-105
                Some(self.get_turn().opposite())
            }
        };
    }
}

pub struct ChessGame {
    pos: ChessPosition,
}

impl IGame for ChessGame {
    type Position = ChessPosition;
    type Move = ChessMove;

    fn new() -> Self {
        Self {
            pos: Self::Position::new(),
        }
    }

    fn new_from_pos(pos: Self::Position) -> Self {
        Self { pos: pos }
    }

    fn get_position(&self) -> Self::Position {
        return self.pos;
    }

    fn play_single_turn(&mut self, player: &mut dyn GamePlayer<Self>) {
        if self.pos.is_over() {
            panic!("game is already over");
        }
        let next_move = player.next_move(&self.pos).unwrap();
        assert!(self.pos.is_valid_move(next_move));
        self.pos = self.pos.get_moved_position(next_move);
    }

    fn play_until_over(
        &mut self,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>) {
        while !self.pos.is_over() {
            self.play_single_turn(match self.pos.get_turn() {
                GameColor::Player1 => player1,
                GameColor::Player2 => player2,
            });
        }
        return (self.pos, self.pos.get_winner());
    }
}
