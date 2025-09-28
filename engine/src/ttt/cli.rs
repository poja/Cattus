use crate::game::player::GamePlayer;
use crate::game::{GameColor, Game};
use crate::ttt::{TttGame, TttMove, TttPosition};
use std::io;

pub struct TttPlayerCmd;
impl GamePlayer<TttGame> for TttPlayerCmd {
    fn next_move(&mut self, pos_history: &[TttPosition]) -> Option<TttMove> {
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
        cli_print_ttt_board(position);

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

pub fn cli_print_ttt_board(pos: &TttPosition) {
    for r in 0..TttGame::BOARD_SIZE {
        let row_characters: Vec<String> = (0..TttGame::BOARD_SIZE)
            .map(|c| match pos.get_tile(r, c) {
                None => String::from("_"),
                Some(GameColor::Player1) => String::from("X"),
                Some(GameColor::Player2) => String::from("O"),
            })
            .collect();
        println!("{}", row_characters.join(" "));
    }
}
