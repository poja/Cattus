use std::cmp::Ordering;
use std::fmt::{self, Display};

use itertools::Itertools;

use crate::game::common::{GameBitboard, GameColor, GameMove, GamePlayer, GamePosition, IGame};
use crate::game::self_play::DataEntry;

pub fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("None"),
        Some(GameColor::Player1) => String::from("X"),
        Some(GameColor::Player2) => String::from("O"),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TttMove {
    idx: u8,
}

impl TttMove {
    pub fn new(r: usize, c: usize) -> Self {
        TttMove::from_idx(r * TttGame::BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        Self { idx: idx as u8 }
    }

    pub fn to_idx(&self) -> usize {
        self.idx as usize
    }

    pub fn row(&self) -> usize {
        self.idx as usize / TttGame::BOARD_SIZE
    }

    pub fn column(&self) -> usize {
        self.idx as usize % TttGame::BOARD_SIZE
    }
}

impl GameMove for TttMove {
    type Game = TttGame;

    fn get_flip(&self) -> Self {
        *self
    }

    fn to_nn_idx(&self) -> usize {
        self.idx as usize
    }
}

impl Display for TttMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.column())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TttBitboard {
    bitmap: u16,
}

impl TttBitboard {
    pub fn get_raw(&self) -> u16 {
        self.bitmap
    }
}

impl GameBitboard for TttBitboard {
    type Game = TttGame;

    fn new() -> Self {
        Self { bitmap: 0 }
    }

    fn new_with_all(val: bool) -> Self {
        Self {
            bitmap: if val { (1u16 << 9) - 1 } else { 0 },
        }
    }

    fn get(&self, idx: usize) -> bool {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        (self.bitmap & (1u16 << idx)) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        assert!(idx < TttGame::BOARD_SIZE * TttGame::BOARD_SIZE);
        if val {
            self.bitmap |= 1u16 << idx;
        } else {
            self.bitmap &= !(1u16 << idx);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TttPosition {
    board_x: TttBitboard,
    board_o: TttBitboard,
    turn: GameColor,
    winner: Option<GameColor>,
}

impl TttPosition {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        if s.chars().count() != TttGame::BOARD_SIZE * TttGame::BOARD_SIZE + 1 {
            panic!("unexpected string length")
        }

        let mut pos = Self::new();
        for (idx, c) in s.chars().enumerate() {
            match idx.cmp(&(TttGame::BOARD_SIZE * TttGame::BOARD_SIZE)) {
                Ordering::Less => match c {
                    'x' => pos.board_x.set(idx, true),
                    'o' => pos.board_o.set(idx, true),
                    '_' => {}
                    _ => panic!("unknown board char: {:?}", c),
                },
                Ordering::Equal => {
                    pos.turn = match c {
                        'x' => GameColor::Player1,
                        'o' => GameColor::Player2,
                        _ => panic!("unknown turn char: {:?}", c),
                    }
                }
                Ordering::Greater => panic!("too many turn chars: {:?}", c),
            }
        }
        pos.check_winner();
        pos
    }

    /* Could lead to invalid board */
    pub fn from_bitboards(board_x: TttBitboard, board_o: TttBitboard, turn: GameColor) -> Self {
        let mut s = Self {
            board_x,
            board_o,
            turn,
            winner: None,
        };
        s.check_winner();
        s
    }

    pub fn pieces_x(&self) -> TttBitboard {
        self.board_x
    }

    pub fn pieces_o(&self) -> TttBitboard {
        self.board_o
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < TttGame::BOARD_SIZE && c < TttGame::BOARD_SIZE);
        let idx = r * TttGame::BOARD_SIZE + c;
        if self.board_x.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_o.get(idx) {
            return Some(GameColor::Player2);
        }
        None
    }

    pub fn make_move(&mut self, m: TttMove) {
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

    pub fn is_valid_move(&self, m: TttMove) -> bool {
        let idx = m.to_idx();
        !self.board_x.get(idx) && !self.board_o.get(idx)
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

impl GamePosition for TttPosition {
    type Game = TttGame;

    fn new() -> Self {
        TttPosition {
            board_x: GameBitboard::new(),
            board_o: GameBitboard::new(),
            turn: GameColor::Player1,
            winner: None,
        }
    }

    fn get_turn(&self) -> GameColor {
        self.turn
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        let mut moves = Vec::new();
        for r in 0..TttGame::BOARD_SIZE {
            for c in 0..TttGame::BOARD_SIZE {
                if self.get_tile(r, c) == None {
                    moves.push(TttMove::new(r, c));
                }
            }
        }
        moves
    }

    fn get_moved_position(&self, m: <Self::Game as IGame>::Move) -> Self {
        assert!(self.is_valid_move(m));
        let mut res = *self;
        res.make_move(m);
        res
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
            winner: self.winner.map(|w| w.opposite()),
        }
    }

    fn print(&self) {
        for r in 0..TttGame::BOARD_SIZE {
            let row_characters: Vec<String> = (0..TttGame::BOARD_SIZE)
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

pub struct TttGame {
    pos: TttPosition,
}

impl TttGame {}

impl IGame for TttGame {
    type Position = TttPosition;
    type Move = TttMove;
    type Bitboard = TttBitboard;
    const BOARD_SIZE: usize = 3;

    fn new() -> Self {
        Self {
            pos: TttPosition::new(),
        }
    }

    fn new_from_pos(pos: Self::Position) -> Self {
        Self { pos }
    }

    fn get_position(&self) -> &Self::Position {
        &self.pos
    }

    fn is_over(&self) -> bool {
        self.pos.is_over()
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        self.pos.get_winner()
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
        (self.pos, self.get_winner())
    }

    fn get_repetition_limit() -> Option<u32> {
        None
    }

    fn produce_transformed_data_entries(entry: DataEntry<Self>) -> Vec<DataEntry<Self>> {
        let transform = |e: &DataEntry<Self>, transform_sq: &dyn Fn(usize) -> usize| {
            let (board_x, board_o) = [e.pos.board_x, e.pos.board_o]
                .iter()
                .map(|b| {
                    let mut bt = TttBitboard::new();
                    for idx in 0..TttGame::BOARD_SIZE * TttGame::BOARD_SIZE {
                        bt.set(transform_sq(idx), b.get(idx));
                    }
                    bt
                })
                .collect_tuple()
                .unwrap();
            let pos = TttPosition::from_bitboards(board_x, board_o, e.pos.get_turn());

            let probs = e
                .probs
                .iter()
                .map(|(m, p)| (TttMove::from_idx(transform_sq(m.to_idx())), *p))
                .collect_vec();

            let winner = e.winner;
            DataEntry { pos, probs, winner }
        };

        let rows_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = TttGame::BOARD_SIZE - 1 - r;
                let ct = c;
                rt * TttGame::BOARD_SIZE + ct
            })
        };
        let columns_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = r;
                let ct = TttGame::BOARD_SIZE - 1 - c;
                rt * TttGame::BOARD_SIZE + ct
            })
        };
        let diagonal_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = c;
                let ct = r;
                rt * TttGame::BOARD_SIZE + ct
            })
        };

        /*
         * Use all combination of the basic transforms:
         * original
         * rows mirror
         * columns mirror
         * diagonal mirror
         * rows + columns = rotate 180
         * rows + diagonal = rotate 90
         * columns + diagonal = rotate 90 (other direction)
         * row + columns + diagonal = other diagonal mirror
         */
        let mut entries = vec![entry];
        entries.extend(entries.iter().map(rows_mirror).collect_vec());
        entries.extend(entries.iter().map(columns_mirror).collect_vec());
        entries.extend(entries.iter().map(diagonal_mirror).collect_vec());
        entries
    }
}
