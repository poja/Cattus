use rand::Rng;
use std::collections::HashSet;

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

const GAME_SIZE: usize = 11;
type Location = (usize, usize);

pub struct HexState {
    /// The board should be imagined in 2D like so:
    /// The board is a rhombus, slanted right. So, board[0][GAME_SIZE - 1] is the "top right end",
    /// also called the "top end" of the board, and board[GAME_SIZE - 1][0] is the "bottom end".
    /// Red tries to move left-right and blue tries to move top-bottom.
    pub board: [[Hexagon; GAME_SIZE]; GAME_SIZE],
    pub turn: Color,
    pub is_over: bool,
    pub winner: Option<Color>,
}

impl HexState {
    pub fn new(starting_color: Color) -> Self {
        Self {
            board: [[Hexagon::Empty; GAME_SIZE]; GAME_SIZE],
            turn: starting_color,
            is_over: false,
            winner: Option::None,
        }
    }

    pub fn print(&self) -> () {
        // TODO there's a RUST way to print
        for row_i in 0..GAME_SIZE {
            let row_characters: Vec<String> = self.board[row_i]
                .iter()
                .map(|hex| String::from(hex.char()))
                .collect();
            let spaces = " ".repeat(GAME_SIZE - row_i - 1);
            println!("{}{}", spaces, row_characters.join(" "));
        }
    }

    /// Returns if turn succeeded
    pub fn make_turn(&mut self, loc: Location) -> bool {
        if self.is_over {
            return false;
        }
        if !in_board(loc) {
            return false;
        } else {
            match self.board[loc.0][loc.1] {
                Hexagon::Empty => {
                    self.board[loc.0][loc.1] = Hexagon::Full(self.turn);
                    self.check_if_over();
                    if !self.is_over {
                        self.flip_turn();
                    }
                    return true;
                }
                _ => {
                    return false;
                }
            }
        }
    }

    fn check_if_over(&mut self) -> () {
        // TODO move these to be static
        let TOP: HashSet<Location> = (0..GAME_SIZE).into_iter().map(|x| (0, x.clone())).collect();
        let BOTTOM: HashSet<Location> = (0..GAME_SIZE)
            .into_iter()
            .map(|x| (GAME_SIZE - 1, x))
            .collect();
        let LEFT: HashSet<Location> = (0..GAME_SIZE).into_iter().map(|x| (x, 0)).collect();
        let RIGHT: HashSet<Location> = (0..GAME_SIZE)
            .into_iter()
            .map(|x| (x, GAME_SIZE - 1))
            .collect();
        if self.has_path(Color::Red, &LEFT, &RIGHT) {
            self.is_over = true;
            self.winner = Some(Color::Red);
        }
        if self.has_path(Color::Blue, &TOP, &BOTTOM) {
            self.is_over = true;
            self.winner = Some(Color::Blue);
        }
    }

    fn has_path(&self, color: Color, src: &HashSet<Location>, dst: &HashSet<Location>) -> bool {
        let relevant_src: HashSet<Location> = src
            .iter()
            .filter(|&loc| {
                in_board(loc.clone()) && self.board[loc.0][loc.1] == Hexagon::Full(color)
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

    fn flip_turn(&mut self) -> () {
        self.turn = match self.turn {
            Color::Red => Color::Blue,
            Color::Blue => Color::Red,
        }
    }
}

fn in_board(loc: Location) -> bool {
    loc.0 < GAME_SIZE && loc.1 < GAME_SIZE
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
        .filter(|&neighbor| in_board(neighbor))
        .collect()
}

/// If no possible turns are available, I will be infinitely stuck
pub fn make_random_turn(game: &mut HexState) {
    let mut rng = rand::thread_rng();
    loop {
        let i = rng.gen_range(0..GAME_SIZE);
        let j = rng.gen_range(0..GAME_SIZE);
        if game.make_turn((i, j)) {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Color::{Blue, Red};

    #[test]
    fn short_diagonal_wins() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);

        let mut s = HexState {
            board: [
                [r, e, e, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, e, e, r],
            ],
            turn: Blue,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(s.winner == Some(Red));

        s = HexState {
            board: [
                [b, e, e, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, e, e, b],
            ],
            turn: Red,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(s.winner == Some(Blue));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);

        let mut s = HexState {
            board: [
                [e, e, e, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, e, e, r],
            ],
            turn: Blue,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(!s.is_over);

        s = HexState {
            board: [
                [b, e, e, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Red,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(!s.is_over);
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);

        let mut s = HexState {
            board: [
                [e, e, e, e, e, e, e, e, e, e, r],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [r, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Blue,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(!s.is_over);

        s = HexState {
            board: [
                [e, e, e, e, e, e, e, e, e, e, b],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [b, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Red,
            winner: None,
            is_over: false,
        };
        s.check_if_over();
        assert!(!s.is_over);
    }
}
