use rl::chess::chess_game::ChessGame;
use rl::chess::cmd_player::ChessPlayerCmd;
use rl::chess::net::net_stockfish::StockfishNet;
use rl::game::common::{GameColor, GamePosition, IGame};
use rl::game::mcts::MCTSPlayer;

fn color_to_str(c: Option<GameColor>) -> String {
    match c {
        None => String::from("Tie"),
        Some(GameColor::Player1) => String::from("White"),
        Some(GameColor::Player2) => String::from("Black"),
    }
}

fn main() {
    let mut player1 = ChessPlayerCmd::new();
    let value_func = Box::new(StockfishNet::new());
    let mut player2 = MCTSPlayer::new_custom(100000, 1.41421, value_func);
    let mut game = ChessGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player2, &mut player1);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
