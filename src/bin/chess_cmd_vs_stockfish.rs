use std::sync::Arc;

use cattus::chess::chess_game::ChessGame;
use cattus::chess::cmd_player::ChessPlayerCmd;
use cattus::chess::net::net_stockfish::StockfishNet;
use cattus::game::common::{GameColor, GamePosition, IGame};
use cattus::game::mcts::MCTSPlayer;

fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("Tie"),
        Some(GameColor::Player1) => String::from("White"),
        Some(GameColor::Player2) => String::from("Black"),
    }
}

fn main() {
    let mut player1 = ChessPlayerCmd {};
    let value_func = Arc::new(StockfishNet {});
    let mut player2 = MCTSPlayer::new(100000, value_func);
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player2, &mut player1);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
