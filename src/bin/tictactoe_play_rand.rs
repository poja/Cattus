use rl::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition, color_to_str};
use rl::game::common::{GamePosition, IGame};
use rl::hex::simple_players::PlayerRand;

fn main() {
    let position = TicTacToePosition::new();
    let mut player1 = PlayerRand::new();
    let mut player2 = PlayerRand::new();
    let (final_pos, winner) = TicTacToeGame::play_until_over(&position, &mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();

}
