use crate::game_utils::game::{GameColor, GameMove, GamePlayer, GamePosition, IGame};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Hexagon {
    Empty,
    Full(GameColor),
}

impl Hexagon {
    fn char(&self) -> char {
        match self {
            Hexagon::Empty => '.',
            Hexagon::Full(GameColor::Player1) => 'R',
            Hexagon::Full(GameColor::Player2) => 'B',
        }
    }
}

pub const BOARD_SIZE: usize = 11;
pub type Location = (usize, usize);

impl GameMove for Location {
    type Game = HexGame;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexPosition {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted right. So, board[0][BOARD_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE],
    turn: GameColor,

    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with red pieces */
    left_red_reach: [[bool; BOARD_SIZE]; BOARD_SIZE],
    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with blue pieces */
    top_blue_reach: [[bool; BOARD_SIZE]; BOARD_SIZE],
    number_of_empty_tiles: u8,
    winner: Option<GameColor>,
}

impl HexPosition {
    pub fn new(starting_color: GameColor) -> Self {
        Self {
            board: [[Hexagon::Empty; BOARD_SIZE]; BOARD_SIZE],
            turn: starting_color,
            left_red_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            top_blue_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        }
    }
    pub fn from_board(board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE], turn: GameColor) -> Self {
        let mut s = Self {
            board: board,
            turn: turn,
            left_red_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            top_blue_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        };

        let is_reach_begin = |r: usize, c: usize, player: GameColor| match player {
            GameColor::Player1 => c == 0,
            GameColor::Player2 => r == 0,
        };
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                match s.board[r][c] {
                    Hexagon::Full(color) => {
                        s.number_of_empty_tiles -= 1;
                        if is_reach_begin(r, c, color) {
                            s.update_reach(r, c, color);
                        }
                    }
                    Hexagon::Empty => {}
                }
            }
        }
        return s;
    }

    pub fn contains(loc: Location) -> bool {
        loc.0 < BOARD_SIZE && loc.1 < BOARD_SIZE
    }

    pub fn is_valid_move(&self, loc: Location) -> bool {
        return HexPosition::contains(loc) && self.board[loc.0][loc.1] == Hexagon::Empty;
    }

    pub fn get_tile(&self, r: usize, c: usize) -> Hexagon {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        self.board[r][c]
    }

    fn foreach_neighbor<OP: FnMut(usize, usize)>(r: usize, c: usize, mut op: OP) {
        let connection_dirs: [(i8, i8); 6] = [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1)];
        for d in connection_dirs {
            let nr = r as i8 + d.0;
            let nc = c as i8 + d.1;
            if nr < 0 || nr as usize >= BOARD_SIZE || nc < 0 || nc as usize >= BOARD_SIZE {
                continue;
            }
            op(nr as usize, nc as usize);
        }
    }

    fn update_reach(&mut self, r: usize, c: usize, player: GameColor) {
        let reach_map = match player {
            GameColor::Player1 => &mut self.left_red_reach,
            GameColor::Player2 => &mut self.top_blue_reach,
        };
        let is_reach_begin = match player {
            GameColor::Player1 => |_: usize, c: usize| c == 0,
            GameColor::Player2 => |r: usize, _: usize| r == 0,
        };
        let is_reach_end = match player {
            GameColor::Player1 => |_: usize, c: usize| c == BOARD_SIZE - 1,
            GameColor::Player2 => |r: usize, _: usize| r == BOARD_SIZE - 1,
        };

        let mut bfs_rqueue: [u8; BOARD_SIZE * BOARD_SIZE] = [0; BOARD_SIZE * BOARD_SIZE];
        let mut bfs_cqueue: [u8; BOARD_SIZE * BOARD_SIZE] = [0; BOARD_SIZE * BOARD_SIZE];
        let mut bfs_queue_size = 0;

        let mut update_reach = is_reach_begin(r, c);
        HexPosition::foreach_neighbor(r, c, |nr: usize, nc: usize| {
            update_reach = update_reach || reach_map[nr as usize][nc as usize];
        });
        if update_reach {
            bfs_rqueue[0] = r as u8;
            bfs_cqueue[0] = c as u8;
            bfs_queue_size += 1;
            reach_map[r][c] = true;
        }

        while bfs_queue_size > 0 {
            bfs_queue_size -= 1;
            let r = bfs_rqueue[bfs_queue_size] as usize;
            let c = bfs_cqueue[bfs_queue_size] as usize;

            if is_reach_end(r, c) {
                self.winner = Some(player);
            } else {
                HexPosition::foreach_neighbor(r, c, |nr: usize, nc: usize| {
                    if !reach_map[nr][nc] && self.board[nr][nc] == Hexagon::Full(player) {
                        bfs_rqueue[bfs_queue_size] = nr as u8;
                        bfs_cqueue[bfs_queue_size] = nc as u8;
                        bfs_queue_size += 1;
                        reach_map[nr][nc] = true;
                    }
                });
            }
        }
    }

    pub fn make_move(&mut self, r: usize, c: usize) {
        assert!(r < BOARD_SIZE && c < BOARD_SIZE);
        assert!(!self.is_over());
        self.board[r][c] = Hexagon::Full(self.turn);

        self.update_reach(r, c, self.turn);

        self.number_of_empty_tiles -= 1;
        self.turn = self.turn.opposite();
    }

    pub fn print(&self) -> () {
        // TODO there's a RUST way to print
        for row_i in 0..BOARD_SIZE {
            let row_characters: Vec<String> = self.board[row_i]
                .iter()
                .map(|hex| String::from(hex.char()))
                .collect();
            let spaces = " ".repeat(BOARD_SIZE - row_i - 1);
            println!("{}{}", spaces, row_characters.join(" "));
        }
    }
}

impl GamePosition for HexPosition {
    type Game = HexGame;
    fn get_turn(&self) -> GameColor {
        self.turn
    }

    fn get_legal_moves(&self) -> Vec<<Self::Game as IGame>::Move> {
        let mut moves = Vec::new();
        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                if self.board[x][y] == Hexagon::Empty {
                    moves.push((x, y));
                }
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
        res.make_move(m.0, m.1);
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

pub struct HexGame {}

impl IGame for HexGame {
    type Position = HexPosition;
    type Move = Location;

    fn play_until_over(
        pos: &Self::Position,
        player1: &mut dyn GamePlayer<Self>,
        player2: &mut dyn GamePlayer<Self>,
    ) -> (Self::Position, Option<GameColor>) {
        let mut position = pos.clone();

        while !position.is_over() {
            let m = match position.get_turn() {
                GameColor::Player1 => player1.next_move(&position),
                GameColor::Player2 => player2.next_move(&position),
            };
            match m {
                None => {
                    if position.is_over() {
                        break;
                    }
                    eprintln!("player failed to choose a move");
                    return (position, Some(position.get_turn().opposite()));
                }
                Some(next_move) => {
                    assert!(position.is_valid_move(next_move));
                    position.make_move(next_move.0, next_move.1);
                }
            }
        }
        return (position, position.get_winner());
    }
}

fn location_neighbors(loc: Location) -> Vec<Location> {
    // same as neighbors in a 2d space but without (+1, -1) and (-1, +1)
    let mut candidates: Vec<Location> = vec![
        (loc.0, loc.1 + 1),
        (loc.0 + 1, loc.1),
        (loc.0 + 1, loc.1 + 1),
    ];
    if loc.0 > 0 {
        candidates.push((loc.0 - 1, loc.1));
    }
    if loc.1 > 0 {
        candidates.push((loc.0, loc.1 - 1));
    }
    if loc.0 > 0 && loc.1 > 0 {
        candidates.push((loc.0 - 1, loc.1 - 1));
    }
    candidates
        .into_iter()
        .filter(|&neighbor| HexPosition::contains(neighbor))
        .collect()
}
