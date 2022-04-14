mod game;
mod hex_game;
mod hex_test;
mod mcts;
mod simple_players;
use game::{GameColor, IGame};
use hex_game::{HexGame, HexPosition};
// use simple_players::HexPlayerCmd;
use std::time::Instant;

fn main() {
    // let mut player1 = PlayerRand::new();
    let mut player1 = mcts::MCTSPlayer::new_custom(100, (2 as f32).sqrt());
    // let mut player2 = HexPlayerCmd::new();
    let mut player2 = mcts::MCTSPlayer::new_custom(100, (2 as f32).sqrt());

    println!("Launching game...");
    let start = Instant::now();
    let (final_pos, winner) = HexGame::play_until_over(
        &HexPosition::new(GameColor::Player1),
        &mut player1,
        &mut player2,
    );
    let duration = start.elapsed();
    println!("Game duration was: {:?}", duration);

    println!("This is the board:");
    final_pos.print();
    match winner {
        Some(color) => {
            println!("The winner is: {:?}", color)
        }
        None => {
            println!("The game ended in draw")
        }
    }
}
