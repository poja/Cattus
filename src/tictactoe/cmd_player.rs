use crate::game::common::GamePlayer;
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToeMove, TicTacToePosition};
use std::io;

pub struct TttPlayerCmd {}

impl TttPlayerCmd {
    pub fn new() -> Self {
        Self {}
    }
}

impl GamePlayer<TicTacToeGame> for TttPlayerCmd {
    fn next_move(&mut self, position: &TicTacToePosition) -> Option<TicTacToeMove> {
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

            let move_ = TicTacToeMove::new(r as u8, c as u8);
            if position.is_valid_move(move_) {
                return Some(move_);
            }
            println!("invalid move");
        }
    }
}
