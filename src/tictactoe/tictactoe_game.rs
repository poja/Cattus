use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame};

pub const BOARD_SIZE: u8 = 3;

pub fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("None"),
        Some(GameColor::Player1) => String::from("X"),
        Some(GameColor::Player2) => String::from("O"),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TicTacToeMove {
    idx: u8,
}

impl TicTacToeMove {
    pub fn new(r: u8, c: u8) -> Self {
        TicTacToeMove::from_idx(r * BOARD_SIZE + c)
    }

    pub fn from_idx(idx: u8) -> Self {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        Self { idx: idx }
    }

    pub fn to_idx(&self) -> u8 {
        self.idx as u8
    }

    pub fn row(&self) -> u8 {
        self.idx / BOARD_SIZE
    }

    pub fn column(&self) -> u8 {
        self.idx % BOARD_SIZE
    }
}

impl GameMove for TicTacToeMove {
    type Game = TicTacToeGame;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bitboard {
    bitmap: u16,
}

impl Bitboard {
    pub fn new() -> Self {
        Self { bitmap: 0 }
    }
    pub fn get_raw(&self) -> u16 {
        self.bitmap
    }

    pub fn get(&self, idx: u8) -> bool {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        return (self.bitmap & (1u16 << idx)) != 0;
    }

    pub fn set(&mut self, idx: u8, val: bool) {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        if val {
            self.bitmap |= 1u16 << idx;
        } else {
            self.bitmap &= !(1u16 << idx);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TicTacToePosition {
    board_x: Bitboard,
    board_y: Bitboard,
    turn: GameColor,
    winner: Option<GameColor>,
    num_empty_tiles: u8,
}

impl TicTacToePosition {
    pub fn get_tile(&self, r: u8, c: u8) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        let idx = r * BOARD_SIZE + c;
        if self.board_x.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_y.get(idx) {
            return Some(GameColor::Player2);
        }
        return None;
    }

    pub fn make_move(&mut self, m: TicTacToeMove) {
        assert!(self.is_valid_move(m));
        assert!(!self.is_over());

        match self.turn {
            GameColor::Player1 => &mut self.board_x,
            GameColor::Player2 => &mut self.board_y,
        }
        .set(m.to_idx(), true);

        self.num_empty_tiles -= 1;
        self.turn = self.turn.opposite();
        self.check_winner();
    }

    pub fn is_valid_move(&self, m: TicTacToeMove) -> bool {
        let idx = m.to_idx();
        return !self.board_x.get(idx) && !self.board_y.get(idx);
    }

    pub fn check_winner(&mut self) {
        let winning_sequences = vec![
            0b111000000, // row 1
            0b000111000, // row 2
            0b000000111, // row 3
            0b100100100, // col 1
            0b010010010, // col 2
            0b001001001, // col 3
            0b100010001, // dial 1
            0b001010100, // dial 2
        ];

        for winning_sequence in winning_sequences {
            if (self.board_x.get_raw() & winning_sequence) == winning_sequence {
                self.winner = Some(GameColor::Player1);
                return;
            }
            if (self.board_y.get_raw() & winning_sequence) == winning_sequence {
                self.winner = Some(GameColor::Player2);
                return;
            }
        }
        self.winner = None;
    }

    pub fn flip_of(pos: &TicTacToePosition) -> Self {
        Self {
            board_x: pos.board_y,
            board_y: pos.board_x,
            turn: pos.turn.opposite(),
            num_empty_tiles: pos.num_empty_tiles,
            winner: match pos.winner {
                Some(w) => Some(w.opposite()),
                None => None,
            },
        }
    }

    pub fn print(&self) -> () {
        for r in 0..BOARD_SIZE {
            let row_characters: Vec<String> = (0..BOARD_SIZE)
                .map(|c| match self.get_tile(r, c) {
                    None => String::from("_"),
                    Some(GameColor::Player1) => String::from("X"),
                    Some(GameColor::Player2) => String::from("O"),
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
            board_x: Bitboard::new(),
            board_y: Bitboard::new(),
            turn: GameColor::Player1,
            winner: None,
            num_empty_tiles: BOARD_SIZE * BOARD_SIZE,
        }
    }

    fn get_turn(&self) -> GameColor {
        self.turn
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        let mut moves = Vec::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.get_tile(r, c) == None {
                    moves.push(TicTacToeMove::new(r, c));
                }
            }
        }
        return moves;
    }

    fn get_moved_position(
        &self,
        m: <Self::Game as IGame>::Move,
    ) -> <Self::Game as IGame>::Position {
        // TODO this is duplicated from hex_game
        assert!(self.is_valid_move(m));
        let mut res = self.clone();
        res.make_move(m);
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
        self.pos.make_move(next_move);
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
