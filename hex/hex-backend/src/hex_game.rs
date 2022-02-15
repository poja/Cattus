use cached::proc_macro::cached;

use std::collections::HashSet;


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
    pub board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE],
    pub turn: Color,
}

impl HexPosition {
    pub fn new(starting_color: Color) -> Self {
        Self {
            board: [[Hexagon::Empty; BOARD_SIZE]; BOARD_SIZE],
            turn: starting_color,
        }
    }

    pub fn get_turn(&self) -> Color {
        self.turn
    }

    pub fn flip_turn(&mut self) -> () {
        self.turn = self.turn.opposite();
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

    pub fn make_move(&mut self, x: usize, y: usize) {
        assert!(x < BOARD_SIZE && y < BOARD_SIZE);
        self.board[x][y] = Hexagon::Full(self.turn);
        self.flip_turn();
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

    pub fn get_winner(&self) -> (bool, Option<Color>) {
       
        if self.has_path(Color::Red, &left_edge(), &right_edge()) {
            return (true, Some(Color::Red));
        } else if self.has_path(Color::Blue, &top_edge(), &bottom_edge()) {
            return (true, Some(Color::Blue));
        } else {
            for x in 0..BOARD_SIZE {
                for y in 0..BOARD_SIZE {
                    if self.board[x][y] == Hexagon::Empty {
                        return (false, None);
                    }
                }
            }
            return (true, None);
        }
    }

    fn has_path(&self, color: Color, src: &HashSet<Location>, dst: &HashSet<Location>) -> bool {
        let relevant_src: HashSet<Location> = src
            .iter()
            .filter(|&loc| {
                HexPosition::contains(*loc) && self.board[loc.0][loc.1] == Hexagon::Full(color)
            })
            .cloned()
            .collect();

        // BFS
        let mut seen = relevant_src.clone();
        let mut worklist = seen.clone();
        while !worklist.is_empty() {
            // pop from worklist
            let loc = worklist.iter().next().cloned().unwrap();
            worklist.remove(&loc);

            for neighbor in location_neighbors(loc) {
                let neighbor_hexagon: Hexagon = self.board[neighbor.0][neighbor.1];
                if neighbor_hexagon != Hexagon::Full(color) {
                    continue;
                }
                if dst.contains(&neighbor) {
                    return true;
                }
                if !seen.contains(&neighbor) {
                    seen.insert(neighbor);
                    worklist.insert(neighbor);
                }
            }
        }
        return false;
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

#[cached]
fn top_edge() -> HashSet<Location> {
    (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (0, x.clone()))
            .collect()
}

#[cached]
fn bottom_edge() -> HashSet<Location> {
    (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (BOARD_SIZE - 1, x))
            .collect()
}

#[cached]
fn left_edge() -> HashSet<Location> {
    (0..BOARD_SIZE).into_iter().map(|x| (x, 0)).collect()
}

#[cached]
fn right_edge() -> HashSet<Location> {
    (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (x, BOARD_SIZE - 1))
            .collect()
}



pub trait HexPlayer {
    fn next_move(&mut self, position: &HexPosition) -> Location;
}

pub struct HexGame<'a> {
    pub position: HexPosition,
    pub is_over: bool,
    pub winner: Option<Color>,

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
        let mut n = Self {
            position: starting_position.clone(),
            is_over: false,
            winner: Option::None,
            player_red: player_red,
            player_blue: player_blue,
        };
        n.check_if_over();
        return n;
    }

    /// Returns if turn succeeded
    pub fn play_next_move(&mut self) -> bool {
        if self.is_over {
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
        self.check_if_over();
        return true;
    }

    pub fn play_until_over(&mut self) -> Option<Color> {
        while !self.is_over {
            self.play_next_move();
        }
        return self.winner;
    }

    pub fn check_if_over(&mut self) -> () {
        let win_status = self.position.get_winner();
        self.is_over = win_status.0;
        self.winner = win_status.1;
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
