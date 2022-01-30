mod hex_game;
mod hex_test;

use hex_game::{Color, HexGame, HexPlayerCmd, HexPlayerRand, HexPosition};

fn main() {
    let player1 = HexPlayerRand::new();
    // let player2 = HexPlayerRand::new();
    let player2 = HexPlayerCmd::new();
    let mut game = HexGame::new(Color::Red, &player1, &player2);
    game.play_until_over();

    println!("This is the board:");
    game.position.print();
    println!("The winner is: {:?}", game.winner.unwrap())
}
