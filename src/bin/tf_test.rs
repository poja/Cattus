use rl::network::simple_network;
use rl::hex::hex_game;
use rl::game_utils::game;

fn main() {
    let net = simple_network::SimpleNetwork::new();
    let pos = hex_game::HexPosition::new_with_starting_color(game::GameColor::Player1);
    let score = net.make_game_prediction(&pos);
    println!("score: {:?}", score);
}
