mod utils_test;
mod hex_game;
mod hex_test;
mod utils;

use hex_game::{Color, HexGame, HexPlayerCmd, HexPlayerRand, HexPosition};

fn main() {
    let mut player1 = HexPlayerRand::new();
    // let mut player2 = HexPlayerRand::new();
    let mut player2 = HexPlayerRand::new();

    let mut game = HexGame::new(Color::Red, &mut player1, &mut player2);
    game.play_until_over();

    println!("This is the board:");
    game.position.print();
    println!("The winner is: {:?}", game.winner.unwrap())
}
