use cattus::chess::chess_game::ChessGame;
use cattus::chess::cmd_player::ChessPlayerCmd;
use cattus::chess::net::net_trivial::TrivialNet;
use cattus::game::common::{GameColor, GamePosition, IGame};
use cattus::game::mcts::{MctsParams, MctsPlayer};
use std::sync::Arc;

fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("Tie"),
        Some(GameColor::Player1) => String::from("White"),
        Some(GameColor::Player2) => String::from("Black"),
    }
}

fn main() {
    cattus::util::init_globals(None);

    let mut player1 = ChessPlayerCmd;
    let value_func = Arc::new(TrivialNet);
    let mut player2 = MctsPlayer::new(MctsParams::new(10000, value_func));
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player2, &mut player1);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
