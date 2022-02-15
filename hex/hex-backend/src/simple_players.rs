use std::io;

use rand::Rng;

use crate::hex_game::{HexPlayer, HexPosition, Location, BOARD_SIZE};

pub struct HexPlayerRand {}

impl HexPlayerRand {
    pub fn new() -> Self {
        Self {}
    }
}

impl HexPlayer for HexPlayerRand {
    fn next_move(&mut self, position: &HexPosition) -> Location {
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

impl HexPlayer for HexPlayerCmd {
    fn next_move(&mut self, position: &HexPosition) -> Location {
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
                return (x, y);
            }
            println!("invalid move");
        }
    }
}
