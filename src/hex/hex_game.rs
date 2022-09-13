use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame};

pub const BOARD_SIZE: u8 = 11;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HexMove {
    idx: u8,
}

impl HexMove {
    pub fn new(r: u8, c: u8) -> Self {
        return HexMove::from_idx(r * BOARD_SIZE + c);
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

impl GameMove for HexMove {
    type Game = HexGame;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bitboard {
    bitmap: u128,
}

impl Bitboard {
    pub fn new() -> Self {
        Self { bitmap: 0 }
    }
    pub fn get_raw(&self) -> u128 {
        self.bitmap
    }

    pub fn get(&self, idx: u8) -> bool {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        return (self.bitmap & (1u128 << idx)) != 0;
    }

    pub fn set(&mut self, idx: u8, val: bool) {
        assert!(idx < BOARD_SIZE * BOARD_SIZE);
        if val {
            self.bitmap |= 1u128 << idx;
        } else {
            self.bitmap &= !(1u128 << idx);
        }
    }

    fn flip(&self) -> Self {
        let mut f = Bitboard::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let idx = r * BOARD_SIZE + c;
                let idxf = c * BOARD_SIZE + r;
                f.set(idxf, self.get(idx));
            }
        }
        return f;
    }

    fn is_empty(&self) -> bool {
        self.bitmap == 0
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexPosition {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted right. So, board[0][BOARD_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    board_red: Bitboard,
    board_blue: Bitboard,
    turn: GameColor,

    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with red pieces */
    left_red_reach: Bitboard,
    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with blue pieces */
    top_blue_reach: Bitboard,
    number_of_empty_tiles: u8,
    winner: Option<GameColor>,
}

impl HexPosition {
    pub fn new_with_starting_color(starting_color: GameColor) -> Self {
        Self {
            board_red: Bitboard::new(),
            board_blue: Bitboard::new(),
            turn: starting_color,
            left_red_reach: Bitboard::new(),
            top_blue_reach: Bitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        }
    }

    pub fn new_from_board(board_red: Bitboard, board_blue: Bitboard, turn: GameColor) -> Self {
        let mut s = Self {
            board_red: board_red,
            board_blue: board_blue,
            turn: turn,
            left_red_reach: Bitboard::new(),
            top_blue_reach: Bitboard::new(),
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        };

        let is_reach_begin = |r: u8, c: u8, player: GameColor| match player {
            GameColor::Player1 => c == 0,
            GameColor::Player2 => r == 0,
        };
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                match s.get_tile(r, c) {
                    Some(color) => {
                        s.number_of_empty_tiles -= 1;
                        if is_reach_begin(r, c, color) {
                            s.update_reach(r, c, color);
                        }
                    }
                    None => {}
                }
            }
        }
        return s;
    }

    pub fn flip_of(pos: &HexPosition) -> Self {
        Self {
            board_red: pos.board_blue.flip(),
            board_blue: pos.board_red.flip(),
            turn: pos.turn.opposite(),
            left_red_reach: pos.top_blue_reach.flip(),
            top_blue_reach: pos.left_red_reach.flip(),
            number_of_empty_tiles: pos.number_of_empty_tiles,
            winner: match pos.winner {
                Some(w) => Some(w.opposite()),
                None => None,
            },
        }
    }

    pub fn pieces_red(&self) -> Bitboard {
        self.board_red
    }

    pub fn pieces_blue(&self) -> Bitboard {
        self.board_blue
    }

    pub fn is_valid_move(&self, m: HexMove) -> bool {
        let idx = m.to_idx();
        return idx < (BOARD_SIZE * BOARD_SIZE) as u8
            && !self.board_red.get(idx)
            && !self.board_blue.get(idx);
    }

    pub fn get_tile(&self, r: u8, c: u8) -> Option<GameColor> {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        let idx = r * BOARD_SIZE + c;
        if self.board_red.get(idx) {
            return Some(GameColor::Player1);
        }
        if self.board_blue.get(idx) {
            return Some(GameColor::Player2);
        }
        return None;
    }

    fn foreach_neighbor<OP: FnMut(u8, u8)>(r: u8, c: u8, mut op: OP) {
        let connection_dirs: [(i8, i8); 6] = [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1)];
        for d in connection_dirs {
            let nr = r as i8 + d.0;
            let nc = c as i8 + d.1;
            if nr < 0 || nr as u8 >= BOARD_SIZE || nc < 0 || nc as u8 >= BOARD_SIZE {
                continue;
            }
            op(nr as u8, nc as u8);
        }
    }

    fn update_reach(&mut self, r: u8, c: u8, player: GameColor) {
        let board = match player {
            GameColor::Player1 => &self.board_red,
            GameColor::Player2 => &self.board_blue,
        };
        let reach_map = match player {
            GameColor::Player1 => &mut self.left_red_reach,
            GameColor::Player2 => &mut self.top_blue_reach,
        };
        let is_reach_begin = match player {
            GameColor::Player1 => |_: u8, c: u8| c == 0,
            GameColor::Player2 => |r: u8, _: u8| r == 0,
        };
        let is_reach_end = match player {
            GameColor::Player1 => |_: u8, c: u8| c == BOARD_SIZE - 1,
            GameColor::Player2 => |r: u8, _: u8| r == BOARD_SIZE - 1,
        };

        let mut bfs_layer = Bitboard::new();

        let mut update_reach = is_reach_begin(r, c);
        HexPosition::foreach_neighbor(r, c, |nr: u8, nc: u8| {
            let n_idx = nr * BOARD_SIZE + nc;
            update_reach = update_reach || reach_map.get(n_idx);
        });
        if update_reach {
            let idx = r * BOARD_SIZE + c;
            reach_map.set(idx, true);
            bfs_layer.set(idx, true);
        }

        while !bfs_layer.is_empty() {
            let idx = bfs_layer.get_raw().trailing_zeros() as u8;
            bfs_layer.set(idx, false);
            let r = idx / BOARD_SIZE;
            let c = idx % BOARD_SIZE;

            if is_reach_end(r, c) {
                self.winner = Some(player);
            } else {
                HexPosition::foreach_neighbor(r, c, |nr: u8, nc: u8| {
                    let n_idx = nr * BOARD_SIZE + nc;
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

    pub fn print(&self) -> () {
        // TODO there's a RUST way to print
        for r in 0..BOARD_SIZE {
            let row_characters: Vec<String> = (0..BOARD_SIZE)
                .map(|c| {
                    String::from(match self.get_tile(r, c) {
                        None => '.',
                        Some(GameColor::Player1) => 'R',
                        Some(GameColor::Player2) => 'B',
                    })
                })
                .collect();
            let spaces = " ".repeat((BOARD_SIZE - r - 1) as usize);
            println!("{}{}", spaces, row_characters.join(" "));
        }
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
        for idx in 0..(BOARD_SIZE * BOARD_SIZE) {
            if !self.board_red.get(idx) && !self.board_blue.get(idx) {
                moves.push(HexMove::from_idx(idx));
            }
        }
        return moves;
    }

    fn get_moved_position(
        &self,
        m: <Self::Game as IGame>::Move,
    ) -> <Self::Game as IGame>::Position {
        assert!(self.is_valid_move(m));
        let mut res = self.clone();
        res.make_move(m);
        return res;
    }

    fn is_over(&self) -> bool {
        self.winner != None || self.number_of_empty_tiles == 0
    }

    fn get_winner(&self) -> Option<GameColor> {
        assert!(self.is_over());
        self.winner
    }
}

pub struct HexGame {
    pos: HexPosition,
}

impl IGame for HexGame {
    type Position = HexPosition;
    type Move = HexMove;

    fn new() -> Self {
        Self {
            pos: HexPosition::new(),
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
