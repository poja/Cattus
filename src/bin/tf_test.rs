use clap::Parser;
use rl::game::common::GameColor;
use rl::hex::hex_game;
use rl::hex::net::scalar_value_net;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model: String,
}

fn main() {
    let args = Args::parse();
    let net = scalar_value_net::ScalarValNet::new(&args.model);
    let pos = hex_game::HexPosition::new_with_starting_color(GameColor::Player1);
    let score = net.evaluate_position(&pos);
    println!("score: {:?}", score);
}
