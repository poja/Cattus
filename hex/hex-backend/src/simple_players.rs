use crate::game::{GamePlayer, GamePosition, IGame};
use crate::hex_game::{HexGame, HexPosition, Location};
use rand::Rng;
use std::io;

pub struct PlayerRand {}

impl PlayerRand {
    pub fn new() -> Self {
        Self {}
    }
}

impl<Game: IGame> GamePlayer<Game> for PlayerRand {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = position.get_legal_moves();
        if moves.len() == 0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        return Some(moves[rng.gen_range(0..moves.len()) as usize]);
    }
}

pub struct HexPlayerCmd {}

impl HexPlayerCmd {
    pub fn new() -> Self {
        Self {}
    }
}

const READ_USIZE_INVALID: usize = usize::MAX;

fn read_usize() -> usize {
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .expect("failed to read input");
    match line.trim().parse::<usize>() {
        Err(e) => {
            println!("invalid number: {}", e);
            return READ_USIZE_INVALID;
        }
        Ok(x) => {
            return x;
        }
    }
}

impl GamePlayer<HexGame> for HexPlayerCmd {
    fn next_move(&mut self, position: &HexPosition) -> Option<Location> {
        println!("Current position:");
        position.print();

        loop {
            println!("Waiting for input move...");
            let x = read_usize();
            if x == READ_USIZE_INVALID {
                continue;
            }
            let y = read_usize();
            if y == READ_USIZE_INVALID {
                continue;
            }

            if position.is_valid_move((x, y)) {
                return Some((x, y));
            }
            println!("invalid move");
        }
    }
}
