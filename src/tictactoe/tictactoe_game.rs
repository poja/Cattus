use crate::game::common::{Bitboard, GameColor, GameMove, GamePlayer, GamePosition, IGame};
use std::fmt::{self, Display};

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
    idx: u8,
}

impl TicTacToeMove {
    pub fn new(r: usize, c: usize) -> Self {
        TicTacToeMove::from_idx(r * BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        Self { idx: idx as u8 }
    }

    pub fn to_idx(&self) -> usize {
        self.idx as usize
    }

    pub fn row(&self) -> usize {
        self.idx as usize / BOARD_SIZE
    }

    pub fn column(&self) -> usize {
        self.idx as usize % BOARD_SIZE
    }
}

impl GameMove for TicTacToeMove {
    type Game = TicTacToeGame;

    fn get_flip(&self) -> Self {
        return *self;
    }

    fn to_nn_idx(&self) -> usize {
        self.idx as usize
    }
}

impl Display for TicTacToeMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.column())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TtoBitboard {
    bitmap: u16,
}

impl TtoBitboard {
    pub fn get_raw(&self) -> u16 {
        self.bitmap
    }
}

impl Bitboard for TtoBitboard {
    fn new() -> Self {
        Self { bitmap: 0 }
    }

    fn new_with_all(val: bool) -> Self {
        Self {
            bitmap: if val { (1u16 << 9) - 1 } else { 0 },
        }
    }

    fn get(&self, idx: usize) -> bool {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        return (self.bitmap & (1u16 << idx)) != 0;
    }

    fn set(&mut self, idx: usize, val: bool) {
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
    board_x: TtoBitboard,
    board_o: TtoBitboard,
    turn: GameColor,
    winner: Option<GameColor>,
}

impl TicTacToePosition {
    pub fn from_str(s: &String) -> Self {
        if s.chars().count() != BOARD_SIZE * BOARD_SIZE + 1 {
            panic!("unexpected string length")
        }

        let mut pos = Self::new();
        for (idx, c) in s.chars().enumerate() {
            if idx < BOARD_SIZE * BOARD_SIZE {
                match c {
                    'x' => pos.board_x.set(idx, true),
                    'o' => pos.board_o.set(idx, true),
                    '_' => {}
                    _ => panic!("unknown board char: {:?}", c),
                }
            } else if idx == BOARD_SIZE * BOARD_SIZE {
                pos.turn = match c {
                    'x' => GameColor::Player1,
                    'o' => GameColor::Player2,
                    _ => panic!("unknown turn char: {:?}", c),
                };
            } else {
                panic!("too many turn chars: {:?}", c);
            }
        }
        pos.check_winner();
        return pos;
    }

    pub fn pieces_x(&self) -> TtoBitboard {
        self.board_x
    }

    pub fn pieces_o(&self) -> TtoBitboard {
        self.board_o
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        let idx = r * BOARD_SIZE + c;
        if self.board_x.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_o.get(idx) {
            return Some(GameColor::Player2);
        }
        return None;
    }

    pub fn make_move(&mut self, m: TicTacToeMove) {
        assert!(self.is_valid_move(m));
        assert!(!self.is_over());

        match self.turn {
            GameColor::Player1 => &mut self.board_x,
            GameColor::Player2 => &mut self.board_o,
        }
        .set(m.to_idx(), true);

        self.turn = self.turn.opposite();
        self.check_winner();
    }

    pub fn is_valid_move(&self, m: TicTacToeMove) -> bool {
        let idx = m.to_idx();
        return !self.board_x.get(idx) && !self.board_o.get(idx);
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
            if (self.board_o.get_raw() & winning_sequence) == winning_sequence {
                self.winner = Some(GameColor::Player2);
                return;
            }
        }
        self.winner = None;
    }
}

impl GamePosition for TicTacToePosition {
    type Game = TicTacToeGame;

    fn new() -> Self {
        TicTacToePosition {
            board_x: Bitboard::new(),
            board_o: Bitboard::new(),
            turn: GameColor::Player1,
            winner: None,
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

    fn get_moved_position(&self, m: <Self::Game as IGame>::Move) -> Self {
        assert!(self.is_valid_move(m));
        let mut res = self.clone();
        res.make_move(m);
        return res;
    }

    fn is_over(&self) -> bool {
        self.winner != None || ((self.board_x.get_raw() | self.board_o.get_raw()) == ((1 << 9) - 1))
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        self.winner
    }

    fn get_flip(&self) -> Self {
        Self {
            board_x: self.board_o,
            board_o: self.board_x,
            turn: self.turn.opposite(),
            winner: match self.winner {
                Some(w) => Some(w.opposite()),
                None => None,
            },
        }
    }

    fn print(&self) -> () {
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

pub struct TicTacToeGame {
    pos: TicTacToePosition,
}

impl TicTacToeGame {}

impl IGame for TicTacToeGame {
    type Position = TicTacToePosition;
    type Move = TicTacToeMove;

    fn new() -> Self {
        Self {
            pos: TicTacToePosition::new(),
        }
    }

    fn new_from_pos(pos: Self::Position) -> Self {
        Self { pos: pos }
    }

    fn get_position(&self) -> &Self::Position {
        return &self.pos;
    }

    fn is_over(&self) -> bool {
        return self.pos.is_over();
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        return self.pos.get_winner();
    }

    fn play_single_turn(&mut self, next_move: Self::Move) {
        assert!(self.pos.is_valid_move(next_move));
        self.pos.make_move(next_move);
    }

    fn play_until_over(
        &mut self,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>) {
        while !self.is_over() {
            let player: &mut dyn GamePlayer<Self> = match self.pos.get_turn() {
                GameColor::Player1 => player1,
                GameColor::Player2 => player2,
            };
            let next_move = player.next_move(&self.pos).unwrap();
            self.play_single_turn(next_move)
        }
        return (self.pos, self.get_winner());
    }

    fn get_repetition_limit() -> Option<u32> {
        return None;
    }
}
