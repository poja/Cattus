use rl::game::common::IGame;
use rl::hex::simple_players::PlayerRand;
use rl::tictactoe::tictactoe_game::{color_to_str, TicTacToeGame};

fn main() {
    let mut player1 = PlayerRand::new();
    let mut player2 = PlayerRand::new();
    let mut game = TicTacToeGame::new();
    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
