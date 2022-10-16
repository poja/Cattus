use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use crate::game::common::{GamePlayer, GamePosition};
use std::io;

pub struct ChessPlayerCmd {}

impl GamePlayer<ChessGame> for ChessPlayerCmd {
    fn next_move(&mut self, position: &ChessPosition) -> Option<ChessMove> {
        let read_cmd_move = || -> Option<ChessMove> {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("failed to read input");

            match ChessMove::from_san(position, line.trim()) {
                Err(e) => {
                    println!("invalid number: {}", e);
                    None
                }
                Ok(x) => Some(x),
            }
        };

        println!("Current position:");
        position.print();

        loop {
            println!("Waiting for input move...");
            let m = match read_cmd_move() {
                None => continue,
                Some(m) => m,
            };

            if position.is_valid_move(m) {
                return Some(m);
            } else {
                println!("invalid move");
            }
        }
    }
}
