use itertools::Itertools;
use std::cmp::Ordering;
use std::fmt::{self, Display};

use crate::game::common::{GameBitboard, GameColor, GameMove, GamePlayer, GamePosition, IGame};
use crate::game::self_play::DataEntry;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexMove {
    idx: u8,
}

impl HexMove {
    pub fn new(r: usize, c: usize) -> Self {
        HexMove::from_idx(r * HexGame::BOARD_SIZE + c)
    }

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx < HexGame::BOARD_SIZE * HexGame::BOARD_SIZE);
        Self { idx: idx as u8 }
    }

    pub fn to_idx(&self) -> usize {
        self.idx as usize
    }

    pub fn row(&self) -> usize {
        self.idx as usize / HexGame::BOARD_SIZE
    }

    pub fn column(&self) -> usize {
        self.idx as usize % HexGame::BOARD_SIZE
    }
}

impl GameMove for HexMove {
    type Game = HexGame;

    fn get_flip(&self) -> Self {
        HexMove::new(self.column(), self.row())
    }

    fn to_nn_idx(&self) -> usize {
        self.idx as usize
    }
}

impl Display for HexMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.column())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexBitboard {
    bitmap: u128,
}

impl HexBitboard {
    pub fn get_raw(&self) -> u128 {
        self.bitmap
    }

    fn flip(&self) -> Self {
        let mut f = HexBitboard::new();
        for r in 0..HexGame::BOARD_SIZE {
            for c in 0..HexGame::BOARD_SIZE {
                let idx = r * HexGame::BOARD_SIZE + c;
                let idxf = c * HexGame::BOARD_SIZE + r;
                f.set(idxf, self.get(idx));
            }
        }
        f
    }

    fn is_empty(&self) -> bool {
        self.bitmap == 0
    }
}

impl GameBitboard for HexBitboard {
    type Game = HexGame;
    fn new() -> Self {
        Self { bitmap: 0 }
    }

    fn new_with_all(val: bool) -> Self {
        Self {
            bitmap: if val { (1u128 << 121) - 1 } else { 0 },
        }
    }

    fn get(&self, idx: usize) -> bool {
        assert!(idx < HexGame::BOARD_SIZE * HexGame::BOARD_SIZE);
        (self.bitmap & (1u128 << idx)) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        assert!(idx < HexGame::BOARD_SIZE * HexGame::BOARD_SIZE);
        if val {
            self.bitmap |= 1u128 << idx;
        } else {
            self.bitmap &= !(1u128 << idx);
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexPosition {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted right. So, board[0][BOARD_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    board_red: HexBitboard,
    board_blue: HexBitboard,
    turn: GameColor,

    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with red pieces */
    left_red_reach: HexBitboard,
    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with blue pieces */
    top_blue_reach: HexBitboard,
    number_of_empty_tiles: u8,
    winner: Option<GameColor>,
}

impl HexPosition {
    pub fn new_with_starting_color(starting_color: GameColor) -> Self {
        Self {
            board_red: HexBitboard::new(),
            board_blue: HexBitboard::new(),
            turn: starting_color,
            left_red_reach: HexBitboard::new(),
            top_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (HexGame::BOARD_SIZE * HexGame::BOARD_SIZE) as u8,
            winner: None,
        }
    }

    pub fn new_from_board(
        board_red: HexBitboard,
        board_blue: HexBitboard,
        turn: GameColor,
    ) -> Self {
        let mut s = Self {
            board_red,
            board_blue,
            turn,
            left_red_reach: HexBitboard::new(),
            top_blue_reach: HexBitboard::new(),
            number_of_empty_tiles: (HexGame::BOARD_SIZE * HexGame::BOARD_SIZE) as u8,
            winner: None,
        };

        let is_reach_begin = |r: usize, c: usize, player: GameColor| match player {
            GameColor::Player1 => c == 0,
            GameColor::Player2 => r == 0,
        };
        for r in 0..HexGame::BOARD_SIZE {
            for c in 0..HexGame::BOARD_SIZE {
                if let Some(color) = s.get_tile(r, c) {
                    s.number_of_empty_tiles -= 1;
                    if is_reach_begin(r, c, color) {
                        s.update_reach(r, c, color);
                    }
                }
            }
        }
        s
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        if s.chars().count() != HexGame::BOARD_SIZE * HexGame::BOARD_SIZE + 1 {
            panic!("unexpected string length")
        }

        let mut board_red = HexBitboard::new();
        let mut board_blue = HexBitboard::new();
        let mut turn = None;
        for (idx, c) in s.chars().enumerate() {
            match idx.cmp(&(HexGame::BOARD_SIZE * HexGame::BOARD_SIZE)) {
                Ordering::Less => match c {
                    'e' => {}
                    'r' => board_red.set(idx, true),
                    'b' => board_blue.set(idx, true),
                    _ => panic!("unknown board char: {:?}", c),
                },
                Ordering::Equal => {
                    turn = Some(match c {
                        'r' => GameColor::Player1,
                        'b' => GameColor::Player2,
                        _ => panic!("unknown turn char: {:?}", c),
                    })
                }
                Ordering::Greater => panic!("Too many chars in position string"),
            }
        }

        Self::new_from_board(board_red, board_blue, turn.unwrap())
    }

    pub fn pieces_red(&self) -> HexBitboard {
        self.board_red
    }

    pub fn pieces_blue(&self) -> HexBitboard {
        self.board_blue
    }

    pub fn is_valid_move(&self, m: HexMove) -> bool {
        let idx = m.to_idx();
        idx < HexGame::BOARD_SIZE * HexGame::BOARD_SIZE
            && !self.board_red.get(idx)
            && !self.board_blue.get(idx)
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Option<GameColor> {
        assert!(r < HexGame::BOARD_SIZE && c < HexGame::BOARD_SIZE);
        let idx = r * HexGame::BOARD_SIZE + c;
        if self.board_red.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_blue.get(idx) {
            return Some(GameColor::Player2);
        }
        None
    }

    fn foreach_neighbor<OP: FnMut(usize, usize)>(r: usize, c: usize, mut op: OP) {
        let connection_dirs: [(i8, i8); 6] = [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1)];
        for (dr, dc) in connection_dirs {
            let nr = r as i8 + dr;
            let nc = c as i8 + dc;
            if nr < 0
                || nr as usize >= HexGame::BOARD_SIZE
                || nc < 0
                || nc as usize >= HexGame::BOARD_SIZE
            {
                continue;
            }
            op(nr as usize, nc as usize);
        }
    }

    fn update_reach(&mut self, r: usize, c: usize, player: GameColor) {
        let board = match player {
            GameColor::Player1 => &self.board_red,
            GameColor::Player2 => &self.board_blue,
        };
        let reach_map = match player {
            GameColor::Player1 => &mut self.left_red_reach,
            GameColor::Player2 => &mut self.top_blue_reach,
        };
        let is_reach_begin = match player {
            GameColor::Player1 => |_: usize, c: usize| c == 0,
            GameColor::Player2 => |r: usize, _: usize| r == 0,
        };
        let is_reach_end = match player {
            GameColor::Player1 => |_: usize, c: usize| c == HexGame::BOARD_SIZE - 1,
            GameColor::Player2 => |r: usize, _: usize| r == HexGame::BOARD_SIZE - 1,
        };

        let mut bfs_layer = HexBitboard::new();

        let mut update_reach = is_reach_begin(r, c);
        HexPosition::foreach_neighbor(r, c, |nr: usize, nc: usize| {
            let n_idx = nr * HexGame::BOARD_SIZE + nc;
            update_reach = update_reach || reach_map.get(n_idx);
        });
        if update_reach {
            let idx = r * HexGame::BOARD_SIZE + c;
            reach_map.set(idx, true);
            bfs_layer.set(idx, true);
        }

        while !bfs_layer.is_empty() {
            let idx = bfs_layer.get_raw().trailing_zeros() as usize;
            bfs_layer.set(idx, false);
            let r = idx / HexGame::BOARD_SIZE;
            let c = idx % HexGame::BOARD_SIZE;

            if is_reach_end(r, c) {
                self.winner = Some(player);
            } else {
                HexPosition::foreach_neighbor(r, c, |nr: usize, nc: usize| {
                    let n_idx = nr * HexGame::BOARD_SIZE + nc;
                    if !reach_map.get(n_idx) && board.get(n_idx) {
                        reach_map.set(n_idx, true);
                        bfs_layer.set(n_idx, true);
                    }
                });
            }
        }
    }

    pub fn make_move(&mut self, m: HexMove) {
        assert!(self.is_valid_move(m));
        assert!(!self.is_over());

        match self.turn {
            GameColor::Player1 => &mut self.board_red,
            GameColor::Player2 => &mut self.board_blue,
        }
        .set(m.to_idx(), true);

        self.update_reach(m.row(), m.column(), self.turn);

        self.number_of_empty_tiles -= 1;
        self.turn = self.turn.opposite();
    }
}

impl GamePosition for HexPosition {
    type Game = HexGame;
    fn new() -> Self {
        HexPosition::new_with_starting_color(GameColor::Player1)
    }
    fn get_turn(&self) -> GameColor {
        self.turn
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        let mut moves = Vec::new();
        for idx in 0..(HexGame::BOARD_SIZE * HexGame::BOARD_SIZE) {
            if !self.board_red.get(idx) && !self.board_blue.get(idx) {
                moves.push(HexMove::from_idx(idx));
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
        self.winner != None || self.number_of_empty_tiles == 0
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        self.winner
    }

    fn get_flip(&self) -> Self {
        Self {
            board_red: self.board_blue.flip(),
            board_blue: self.board_red.flip(),
            turn: self.turn.opposite(),
            left_red_reach: self.top_blue_reach.flip(),
            top_blue_reach: self.left_red_reach.flip(),
            number_of_empty_tiles: self.number_of_empty_tiles,
            winner: self.winner.map(|w| w.opposite()),
        }
    }

    fn print(&self) {
        for r in 0..HexGame::BOARD_SIZE {
            let row_characters: Vec<String> = (0..HexGame::BOARD_SIZE)
                .map(|c| {
                    String::from(match self.get_tile(r, c) {
                        None => '.',
                        Some(GameColor::Player1) => 'R',
                        Some(GameColor::Player2) => 'B',
                    })
                })
                .collect();
            let spaces = " ".repeat(HexGame::BOARD_SIZE - r - 1);
            println!("{}{}", spaces, row_characters.join(" "));
        }
    }
}

pub struct HexGame {
    pos: HexPosition,
}

impl IGame for HexGame {
    type Position = HexPosition;
    type Move = HexMove;
    type Bitboard = HexBitboard;
    const BOARD_SIZE: usize = 11;

    fn new() -> Self {
        Self {
            pos: HexPosition::new(),
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
            let (board_red, board_blue) = [e.pos.board_red, e.pos.board_blue]
                .iter()
                .map(|b| {
                    let mut bt = HexBitboard::new();
                    for idx in 0..HexGame::BOARD_SIZE * HexGame::BOARD_SIZE {
                        bt.set(transform_sq(idx), b.get(idx));
                    }
                    bt
                })
                .collect_tuple()
                .unwrap();
            let pos = HexPosition::new_from_board(board_red, board_blue, e.pos.get_turn());

            let probs = e
                .probs
                .iter()
                .map(|(m, p)| (HexMove::from_idx(transform_sq(m.to_idx())), *p))
                .collect_vec();

            let winner = e.winner;
            DataEntry { pos, probs, winner }
        };

        let rotate_180 = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / HexGame::BOARD_SIZE, idx % HexGame::BOARD_SIZE);
                let rt = HexGame::BOARD_SIZE - 1 - r;
                let ct = HexGame::BOARD_SIZE - 1 - c;
                rt * HexGame::BOARD_SIZE + ct
            })
        };

        let mut entries = vec![entry];
        entries.extend(entries.iter().map(rotate_180).collect_vec());
        entries
    }
}
