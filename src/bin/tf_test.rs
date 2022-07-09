use rl::hex::network::simple;
use rl::hex::hex_game;
use rl::game_utils::game;
use clap::Parser;


#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model: String,
}

fn main() {
    let args = Args::parse();
    let net = simple::SimpleNetwork::new(args.model);
    let pos = hex_game::HexPosition::new_with_starting_color(game::GameColor::Player1);
    let score = net.evaluate_position(&pos);
    println!("score: {:?}", score);
}
