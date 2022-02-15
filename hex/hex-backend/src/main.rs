mod hex_game;
mod simple_players;
mod mcts;

use hex_game::{Color, HexGame};

fn main() {
    // let mut player1 = HexPlayerRand::new();
    let mut player1 = mcts::MCTSPlayer::with_simulations_per_move(50);
    // let mut player2 = HexPlayerRand::new();
    let mut player2 = mcts::MCTSPlayer::with_simulations_per_move(100);

    let mut game = HexGame::new(Color::Red, &mut player1, &mut player2);
    game.play_until_over();

    println!("This is the board:");
    game.position.print();
    println!("The winner is: {:?}", game.winner.unwrap())
}
