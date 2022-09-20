use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame};
use chess;
use itertools::Itertools;

pub const BOARD_SIZE: usize = 8;

fn err_to_str(err: chess::Error) -> String {
    match err {
        chess::Error::InvalidFen { fen } => String::from("invalid fen: ") + &fen,
        chess::Error::InvalidBoard => String::from("invalid board"),
        chess::Error::InvalidSquare => String::from("invalid square"),
        chess::Error::InvalidSanMove => String::from("invalid SAN move"),
        chess::Error::InvalidUciMove => String::from("invalid UCI move"),
        chess::Error::InvalidRank => String::from("invalid rank"),
        chess::Error::InvalidFile => String::from("invalid file"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChessMove {
    m: chess::ChessMove,
}
impl ChessMove {
    pub fn new(m: chess::ChessMove) -> Self {
        Self { m: m }
    }
    pub fn from_str(pos: &ChessPosition, move_text: &str) -> Result<Self, String> {
        return match chess::ChessMove::from_san(&pos.board, move_text) {
            Ok(m) => Ok(Self::new(m)),
            Err(e) => Err(err_to_str(e)),
        };
    }
}
impl GameMove for ChessMove {
    type Game = ChessGame;

    fn to_nn_idx(&self) -> usize {
        self.m.get_source().to_index() * 64 + self.m.get_dest().to_index()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ChessPosition {
    board: chess::Board,
    fifth_rule_count: u8,
}

impl ChessPosition {
    fn new_from_board(board: chess::Board) -> Self {
        Self {
            board: board,
            fifth_rule_count: 0,
        }
    }

    // pub fn flip_of(pos: &ChessPosition) -> Self {}

    pub fn is_valid_move(&self, m: ChessMove) -> bool {
        !self.is_over() && self.board.legal(m.m)
    }

    // pub fn get_tile(&self, r: usize, c: usize) -> Hexagon {
    //     assert!(r < BOARD_SIZE && c < BOARD_SIZE);
    //     self.board[r][c]
    // }

    pub fn print(&self) {
        let square_str = |rank, file| -> String {
            let square = chess::Square::make_square(
                chess::Rank::from_index(rank),
                chess::File::from_index(file),
            );
            match self.board.piece_on(square) {
                Some(piece) => piece.to_string(self.board.color_on(square).unwrap()),
                None => "_".to_string(),
            }
        };

        for rank in (0..BOARD_SIZE).rev() {
            let row_chars: Vec<String> =
                (0..BOARD_SIZE).map(|file| square_str(rank, file)).collect();
            println!("{} | {}", (rank + 1), row_chars.join(" "));
        }

        let files = vec!["A", "B", "C", "D", "E", "F", "G", "H"];
        let files_indices: Vec<String> = (0..BOARD_SIZE)
            .map(|file| files[file].to_string())
            .collect();
        println!("    {}", "-".repeat(BOARD_SIZE * 2 - 1));
        println!("    {}", files_indices.join(" "));
    }
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
        Self::new_from_board(chess::Board::default())
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
        assert!(self.is_valid_move(m));

        let mut next_board = ChessPosition::new_from_board(self.board.make_move_new(m.m));

        let piece = self.board.piece_on(m.m.get_source());
        let is_pawn = piece.is_some() && piece.unwrap() == chess::Piece::Pawn;
        let is_atk = self.board.piece_on(m.m.get_dest()).is_some();

        next_board.fifth_rule_count = if is_pawn || is_atk {
            0
        } else if self.get_turn() == GameColor::Player1 {
            /* The move count is incremented only once per white+black move */
            self.fifth_rule_count + 1
        } else {
            self.fifth_rule_count
        };

        return next_board;
    }

    fn is_over(&self) -> bool {
        self.board.status() != chess::BoardStatus::Ongoing || self.fifth_rule_count >= 50
    }

    fn get_winner(&self) -> Option<GameColor> {
        return match self.board.status() {
            chess::BoardStatus::Ongoing => {
                if self.fifth_rule_count >= 50 {
                    return None;
                } else {
                    panic!("Game is not over")
                }
            }
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
