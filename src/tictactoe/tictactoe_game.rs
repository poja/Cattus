use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame};

pub const BOARD_SIZE: usize = 3;

pub fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("None"),
        Some(GameColor::Player1) => String::from("X"),
        Some(GameColor::Player2) => String::from("O"),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TicTacToeMove {
    pub cell: (usize, usize),
}

impl TicTacToeMove {
    pub fn new(x: usize, y: usize) -> Self {
        Self { cell: (x, y) }
    }
}

impl GameMove for TicTacToeMove {
    type Game = TicTacToeGame;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TicTacToePosition {
    board: [[usize; BOARD_SIZE]; BOARD_SIZE], // 0 is empty, 1 is player1, 2 is player2
    turn: GameColor,
    winner: Option<GameColor>,
    num_empty_tiles: usize,
}

impl TicTacToePosition {
    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        return match self.board[r][c] {
            0 => None,
            1 => Some(GameColor::Player1),
            2 => Some(GameColor::Player2),
            _ => panic!("unexpected value"),
        };
    }

    pub fn make_move(&mut self, r: usize, c: usize) {
        assert!(r <= 2 && c <= 2);
        assert!(!self.is_over());

        self.board[r][c] = if self.turn == GameColor::Player1 {
            1
        } else {
            2
        };
        self.num_empty_tiles -= 1;
        self.turn = self.turn.opposite();

        self.check_winner();
    }

    pub fn is_valid_move(&self, m: TicTacToeMove) -> bool {
        self.board[m.cell.0][m.cell.1] == 0
    }

    pub fn check_winner(&mut self) {
        let is_sequence = |x, y, z| x != 0 && x == y && y == z;
        let num_to_some_player = |n| {
            if n == 1 {
                Some(GameColor::Player1)
            } else {
                Some(GameColor::Player2)
            }
        };
        for row_i in 0..=2 {
            let r = self.board[row_i];
            if is_sequence(r[0], r[1], r[2]) {
                self.winner = num_to_some_player(r[0]);
            }
        }
        let b = self.board;
        for col_i in 0..=2 {
            if is_sequence(b[0][col_i], b[1][col_i], b[2][col_i]) {
                self.winner = num_to_some_player(b[0][col_i]);
            }
        }
        if is_sequence(b[0][0], b[1][1], b[2][2]) {
            self.winner = num_to_some_player(b[0][0]);
        };
        if is_sequence(b[0][2], b[1][1], b[2][0]) {
            self.winner = num_to_some_player(b[0][2]);
        };
    }

    pub fn print(&self) -> () {
        // TODO there's a RUST way to print
        for row_i in 0..=2 {
            let row_characters: Vec<String> = self.board[row_i]
                .iter()
                .map(|symbol| match symbol {
                    0 => String::from("_"),
                    1 => String::from("X"),
                    2 => String::from("O"),
                    _ => panic!("Board is corrupt."),
                })
                .collect();
            println!("{}", row_characters.join(" "));
        }
    }
}

impl GamePosition for TicTacToePosition {
    type Game = TicTacToeGame;

    fn new() -> Self {
        TicTacToePosition {
            board: [[0; 3]; 3],
            turn: GameColor::Player1,
            winner: None,
            num_empty_tiles: 9,
        }
    }

    fn get_turn(&self) -> GameColor {
        self.turn
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        let mut moves = Vec::new();
        for x in 0..=2 {
            for y in 0..=2 {
                if self.board[x][y] == 0 {
                    moves.push(TicTacToeMove::new(x, y));
                }
            }
        }
        return moves;
    }

    fn get_moved_position(
        &self,
        m: <Self::Game as IGame>::Move,
    ) -> <Self::Game as IGame>::Position {
        // TODO this is duplicated fro hex_game
        assert!(self.is_valid_move(m));
        let mut res = self.clone();
        res.make_move(m.cell.0, m.cell.1);
        return res;
    }

    fn is_over(&self) -> bool {
        self.winner != None || self.num_empty_tiles == 0
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        self.winner
    }
}

pub struct TicTacToeGame {
    pos: TicTacToePosition,
}

impl TicTacToeGame {}

impl IGame for TicTacToeGame {
    type Position = TicTacToePosition;
    type Move = TicTacToeMove;

    // TODO this is duplicated from hex_game

    fn new() -> Self {
        Self {
            pos: TicTacToePosition::new(),
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
        self.pos.make_move(next_move.cell.0, next_move.cell.1);
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
