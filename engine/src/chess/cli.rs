use crate::chess::{ChessGame, ChessMove, ChessPosition};
use crate::game::player::GamePlayer;
use crate::game::Game;
use std::io;

pub struct ChessPlayerCmd;
impl GamePlayer<ChessGame> for ChessPlayerCmd {
    fn next_move(&mut self, pos_history: &[ChessPosition]) -> Option<ChessMove> {
        let position = pos_history.last().unwrap();
        let read_cli_move = || -> Option<ChessMove> {
            let mut line = String::new();
            io::stdin().read_line(&mut line).expect("failed to read input");

            match ChessMove::from_san(position, line.trim()) {
                Err(e) => {
                    println!("invalid number: {}", e);
                    None
                }
                Ok(x) => Some(x),
            }
        };

        println!("Current position:");
        cli_print_chess_board(position);

        loop {
            println!("Waiting for input move...");
            let m = match read_cli_move() {
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

pub fn cli_print_chess_board(pos: &ChessPosition) {
    let square_str = |rank, file| -> String {
        let square = chess::Square::make_square(chess::Rank::from_index(rank), chess::File::from_index(file));
        match pos.board.piece_on(square) {
            Some(piece) => piece.to_string(pos.board.color_on(square).unwrap()),
            None => "·".to_string(),
        }
    };

    for rank in (0..ChessGame::BOARD_SIZE).rev() {
        let row_chars: Vec<String> = (0..ChessGame::BOARD_SIZE).map(|file| square_str(rank, file)).collect();
        println!("{} | {}", (rank + 1), row_chars.join(" "));
    }

    let files = ["A", "B", "C", "D", "E", "F", "G", "H"];
    let files_indices: Vec<String> = (0..ChessGame::BOARD_SIZE).map(|file| files[file].to_string()).collect();
    println!("    {}", "-".repeat(ChessGame::BOARD_SIZE * 2 - 1));
    println!("    {}", files_indices.join(" "));
}
