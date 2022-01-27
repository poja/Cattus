mod hex_game;

use hex_game::{HexState, Color, make_random_turn};

fn main() {
    let mut s = HexState::new(Color::Red);
    while !s.is_over {
        make_random_turn(&mut s);
    }
    println!("This is the board:");
    s.print();
    println!("The winner is: {:?}", s.winner.unwrap())
}
