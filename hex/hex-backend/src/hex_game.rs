use rand::Rng;
use std::collections::HashSet;
use std::io;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    Red,
    Blue,
}

const BOARD_SIZE: usize = 11;
type Location = (usize, usize);

#[derive(Clone)]
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

    pub fn contains(loc: Location) -> bool {
        loc.0 < BOARD_SIZE && loc.1 < BOARD_SIZE
    }

    pub fn is_valid_move(&self, loc: Location) -> bool {
        return HexPosition::contains(loc) && self.board[loc.0][loc.1] == Hexagon::Empty;
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
    fn next_move(&self, position: &HexPosition) -> Location;
}

pub struct HexGame<'a> {
    pub position: HexPosition,
    pub is_over: bool,
    pub winner: Option<Color>,

    player_red: &'a dyn HexPlayer,
    player_blue: &'a dyn HexPlayer,
}

impl<'a> HexGame<'a> {
    pub fn new(
        starting_color: Color,
        player_red: &'a dyn HexPlayer,
        player_blue: &'a dyn HexPlayer,
    ) -> Self {
        let empty_position = HexPosition::new(starting_color);
        return HexGame::from_position(&empty_position, player_red, player_blue);
    }

    pub fn from_position(
        starting_position: &HexPosition,
        player_red: &'a dyn HexPlayer,
        player_blue: &'a dyn HexPlayer,
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
        let next_move = match self.position.turn {
            Color::Red => self.player_red.next_move(&self.position),
            Color::Blue => self.player_blue.next_move(&self.position),
        };
        if !self.position.is_valid_move(next_move) {
            return false;
        }
        self.position.board[next_move.0][next_move.1] = Hexagon::Full(self.position.turn);
        self.check_if_over();
        if !self.is_over {
            self.flip_turn();
        }
        return true;
    }

    pub fn play_until_over(&mut self) -> Option<Color> {
        while !self.is_over {
            self.play_next_move();
        }
        return self.winner;
    }

    pub fn check_if_over(&mut self) -> () {
        // TODO move these to be static
        let top: HashSet<Location> = (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (0, x.clone()))
            .collect();
        let bottom: HashSet<Location> = (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (BOARD_SIZE - 1, x))
            .collect();
        let left: HashSet<Location> = (0..BOARD_SIZE).into_iter().map(|x| (x, 0)).collect();
        let right: HashSet<Location> = (0..BOARD_SIZE)
            .into_iter()
            .map(|x| (x, BOARD_SIZE - 1))
            .collect();
        if self.has_path(Color::Red, &left, &right) {
            self.is_over = true;
            self.winner = Some(Color::Red);
        } else if self.has_path(Color::Blue, &top, &bottom) {
            self.is_over = true;
            self.winner = Some(Color::Blue);
        } else {
            for x in 0..BOARD_SIZE {
                for y in 0..BOARD_SIZE {
                    if self.position.board[x][y] == Hexagon::Empty {
                        return;
                    }
                }
            }
            self.is_over = true;
            self.winner = None;
        }
    }

    fn has_path(&self, color: Color, src: &HashSet<Location>, dst: &HashSet<Location>) -> bool {
        let relevant_src: HashSet<Location> = src
            .iter()
            .filter(|&loc| {
                HexPosition::contains(loc.clone())
                    && self.position.board[loc.0][loc.1] == Hexagon::Full(color)
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
                let neighbor_hexagon: Hexagon = self.position.board[neighbor.0][neighbor.1];
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

    fn flip_turn(&mut self) -> () {
        self.position.turn = match self.position.turn {
            Color::Red => Color::Blue,
            Color::Blue => Color::Red,
        }
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

pub struct HexPlayerRand {}

impl HexPlayerRand {
    pub fn new() -> Self {
        Self {}
    }
}

impl HexPlayer for HexPlayerRand {
    fn next_move(&self, position: &HexPosition) -> Location {
        let mut rng = rand::thread_rng();
        loop {
            let i = rng.gen_range(0..BOARD_SIZE);
            let j = rng.gen_range(0..BOARD_SIZE);
            if position.is_valid_move((i, j)) {
                return (i, j);
            }
        }
    }
}

pub struct HexPlayerCmd {}

impl HexPlayerCmd {
    pub fn new() -> Self {
        Self {}
    }
}

fn read_usize() -> usize {
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .expect("failed to read input");
    match line.trim().parse::<usize>() {
        Err(e) => {
            println!("invalid number: {}", e);
            return 0xffffffff;
        }
        Ok(x) => {
            return x;
        }
    }
}

impl HexPlayer for HexPlayerCmd {
    fn next_move(&self, position: &HexPosition) -> Location {
        println!("Current position:");
        position.print();

        loop {
            println!("Waiting for input move...");
            let x = read_usize();
            if x == 0xffffffff {
                continue;
            }
            let y = read_usize();
            if y == 0xffffffff {
                continue;
            }

            if position.is_valid_move((x, y)) {
                return (x, y);
            }
            println!("invalid move");
        }
    }
}
