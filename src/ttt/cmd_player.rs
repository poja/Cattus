use crate::game::common::{GamePlayer, GamePosition};
use crate::ttt::ttt_game::{TttGame, TttMove, TttPosition};
use std::io;

pub struct TttPlayerCmd {}

impl GamePlayer<TttGame> for TttPlayerCmd {
    fn next_move(&mut self, position: &TttPosition) -> Option<TttMove> {
        let read_usize = || -> Option<usize> {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("failed to read input");
            match line.trim().parse::<usize>() {
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
            let r = match read_usize() {
                None => continue,
                Some(r) => r,
            };
            let c = match read_usize() {
                None => continue,
                Some(c) => c,
            };

            let move_ = TttMove::new(r, c);
            if position.is_valid_move(move_) {
                return Some(move_);
            }
            println!("invalid move");
        }
    }
}
