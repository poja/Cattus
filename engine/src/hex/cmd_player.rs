use std::io;

use crate::game::common::{GamePlayer, GamePosition, IGame};
use crate::hex::hex_game::{HexGame, HexMove};

pub struct HexPlayerCmd;
impl<const BOARD_SIZE: usize> GamePlayer<HexGame<BOARD_SIZE>> for HexPlayerCmd {
    fn next_move(
        &mut self,
        pos_history: &[<HexGame<BOARD_SIZE> as IGame>::Position],
    ) -> Option<<HexGame<BOARD_SIZE> as IGame>::Move> {
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
        let position = pos_history.last().unwrap();
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
            let m = HexMove::new(r, c);

            if position.is_valid_move(m) {
                return Some(m);
            }
            println!("invalid move");
        }
    }
}
