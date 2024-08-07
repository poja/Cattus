use cattus::chess::chess_game::ChessGame;
use cattus::chess::cmd_player::ChessPlayerCmd;
use cattus::game::common::{GameColor, GamePosition, IGame};
use cattus::utils;

fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("Tie"),
        Some(GameColor::Player1) => String::from("White"),
        Some(GameColor::Player2) => String::from("Black"),
    }
}

fn main() {
    utils::init_globals();

    let mut player1 = ChessPlayerCmd {};
    let mut player2 = ChessPlayerCmd {};
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
