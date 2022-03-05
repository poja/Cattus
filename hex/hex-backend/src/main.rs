mod hex_game;
mod hex_test;
mod mcts;
mod simple_players;
use hex_game::{Color, HexGame};
use std::time::Instant;

fn main() {
    // let mut player1 = HexPlayerRand::new();
    let mut player1 = mcts::MCTSPlayer::with_simulations_per_move(50);
    // let mut player2 = HexPlayerRand::new();
    let mut player2 = mcts::MCTSPlayer::with_simulations_per_move(100);

    let mut game = HexGame::new(Color::Red, &mut player1, &mut player2);
    let start = Instant::now();
    let winner = game.play_until_over();
    let duration = start.elapsed();
    println!("Game duration was: {:?}", duration);

    println!("This is the board:");
    game.position.print();
    match winner {
        Some(color) => {
            println!("The winner is: {:?}", color)
        }
        None => {
            println!("The game ended in draw")
        }
    }
}
