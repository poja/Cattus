#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Hexagon {
    Empty,
    Full(Color),
}

impl Hexagon {
    fn char(&self) -> char {
        match self {
            Hexagon::Empty => '.',
            Hexagon::Full(Color::Red) => 'R',
            Hexagon::Full(Color::Blue) => 'B',
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Color {
    Red,
    Blue,
}

impl Color {
    fn opposite(&self) -> Color {
        match self {
            Color::Red => Color::Blue,
            Color::Blue => Color::Red,
        }
    }
}

pub const BOARD_SIZE: usize = 11;
pub type Location = (usize, usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct HexPosition {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted right. So, board[0][BOARD_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[BOARD_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE],
    turn: Color,

    /* bitmap of all the tiles one can reach from the left side of the board stepping only on tiles with red pieces */
    left_red_reach: [[bool; BOARD_SIZE]; BOARD_SIZE],
    /* bitmap of all the tiles one can reach from the top side of the board stepping only on tiles with blue pieces */
    top_blue_reach: [[bool; BOARD_SIZE]; BOARD_SIZE],
    number_of_empty_tiles: u8,
    winner: Option<Color>,
}

impl HexPosition {
    pub fn new(starting_color: Color) -> Self {
        Self {
            board: [[Hexagon::Empty; BOARD_SIZE]; BOARD_SIZE],
            turn: starting_color,
            left_red_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            top_blue_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        }
    }
    pub fn from_board(board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE], turn: Color) -> Self {
        let mut s = Self {
            board: board,
            turn: turn,
            left_red_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            top_blue_reach: [[false; BOARD_SIZE]; BOARD_SIZE],
            number_of_empty_tiles: (BOARD_SIZE * BOARD_SIZE) as u8,
            winner: None,
        };

        let is_reach_begin = |r: usize, c: usize, player: Color| match player {
            Color::Red => c == 0,
            Color::Blue => r == 0,
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

    pub fn get_turn(&self) -> Color {
        self.turn
    }

    pub fn contains(loc: Location) -> bool {
        loc.0 < BOARD_SIZE && loc.1 < BOARD_SIZE
    }

    pub fn is_valid_move(&self, loc: Location) -> bool {
        return HexPosition::contains(loc) && self.board[loc.0][loc.1] == Hexagon::Empty;
    }

    pub fn get_tile(&self, x: usize, y: usize) -> Hexagon {
        assert!(x < BOARD_SIZE && y < BOARD_SIZE);
        self.board[x][y]
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

    fn update_reach(&mut self, r: usize, c: usize, player: Color) {
        let reach_map = match player {
            Color::Red => &mut self.left_red_reach,
            Color::Blue => &mut self.top_blue_reach,
        };
        let is_reach_begin = match player {
            Color::Red => |_: usize, c: usize| c == 0,
            Color::Blue => |r: usize, _: usize| r == 0,
        };
        let is_reach_end = match player {
            Color::Red => |_: usize, c: usize| c == BOARD_SIZE - 1,
            Color::Blue => |r: usize, _: usize| r == BOARD_SIZE - 1,
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

    pub fn get_moved_position(&self, loc: Location) -> HexPosition {
        assert!(self.is_valid_move(loc));
        let mut res = self.clone();
        res.make_move(loc.0, loc.1);
        return res;
    }

    pub fn get_legal_moves(&self) -> Vec<Location> {
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

    pub fn is_over(&self) -> bool {
        self.winner != None || self.number_of_empty_tiles == 0
    }

    pub fn get_winner(&self) -> Option<Color> {
        assert!(self.is_over());
        self.winner
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

pub trait HexPlayer {
    fn next_move(&mut self, position: &HexPosition) -> Location;
}

pub struct HexGame<'a> {
    pub position: HexPosition,

    player_red: &'a mut dyn HexPlayer,
    player_blue: &'a mut dyn HexPlayer,
}

impl<'a> HexGame<'a> {
    pub fn new(
        starting_color: Color,
        player_red: &'a mut dyn HexPlayer,
        player_blue: &'a mut dyn HexPlayer,
    ) -> Self {
        HexGame::from_position(&HexPosition::new(starting_color), player_red, player_blue)
    }

    pub fn from_position(
        starting_position: &HexPosition,
        player_red: &'a mut dyn HexPlayer,
        player_blue: &'a mut dyn HexPlayer,
    ) -> Self {
        Self {
            position: starting_position.clone(),
            player_red: player_red,
            player_blue: player_blue,
        }
    }

    /// Returns if turn succeeded
    pub fn play_next_move(&mut self) -> bool {
        if self.position.is_over() {
            return false;
        }
        let next_move = match self.position.get_turn() {
            Color::Red => self.player_red.next_move(&self.position),
            Color::Blue => self.player_blue.next_move(&self.position),
        };
        if !self.position.is_valid_move(next_move) {
            return false;
        }
        self.position.make_move(next_move.0, next_move.1);
        return true;
    }

    pub fn play_until_over(&mut self) -> Option<Color> {
        while !self.position.is_over() {
            self.play_next_move();
        }
        return self.position.get_winner();
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
