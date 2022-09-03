use crate::game::common::GamePlayer;
use crate::hex::hex_game::{HexGame, HexPosition, Location};
use std::io;

pub struct HexPlayerCmd {}

impl HexPlayerCmd {
    pub fn new() -> Self {
        Self {}
    }
}

impl GamePlayer<HexGame> for HexPlayerCmd {
    fn next_move(&mut self, position: &HexPosition) -> Option<Location> {
        let read_usize = || -> Option<usize> {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("failed to read input");
            match line.trim().parse::<usize>() {
                Err(e) => {
                    println!("invalid number: {}", e);
                    return None;
                }
                Ok(x) => {
                    return Some(x);
                }
            }
        };

        println!("Current position:");
        position.print();

        loop {
            println!("Waiting for input move...");
            let r = match read_usize() {
                None => continue,
                Some(r) => r,
            };
            let c = match read_usize() {
                None => continue,
                Some(c) => c,
            };

            if position.is_valid_move((r, c)) {
                return Some((r, c));
            }
            println!("invalid move");
        }
    }
}
