use std::io;

use crate::game::common::{GameColor, GamePlayer, IGame};
use crate::hex::hex_game::{HexGame, HexMove, HexPosition};

pub struct HexPlayerCmd;
impl<const BOARD_SIZE: usize> GamePlayer<HexGame<BOARD_SIZE>> for HexPlayerCmd {
    fn next_move(
        &mut self,
        pos_history: &[<HexGame<BOARD_SIZE> as IGame>::Position],
    ) -> Option<<HexGame<BOARD_SIZE> as IGame>::Move> {
        let read_usize = || -> Option<usize> {
            let mut line = String::new();
            io::stdin().read_line(&mut line).expect("failed to read input");
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
        cmd_print_hex_board(position);

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

pub fn cmd_print_hex_board<const BOARD_SIZE: usize>(pos: &HexPosition<BOARD_SIZE>) {
    for r in 0..BOARD_SIZE {
        let row_characters: Vec<String> = (0..BOARD_SIZE)
            .map(|c| {
                String::from(match pos.get_tile(r, c) {
                    None => '·',
                    Some(GameColor::Player1) => 'R',
                    Some(GameColor::Player2) => 'B',
                })
            })
            .collect();
        let spaces = " ".repeat(BOARD_SIZE - r - 1);
        println!("{}{}", spaces, row_characters.join(" "));
    }
}
