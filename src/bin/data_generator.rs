use clap::Parser;
use rl::game_utils::train;
use rl::game_utils::{game, mcts};
use rl::hex::hex_game;
use rl::network::simple_network;

// #[derive(Parser, Debug)]
// #[clap(about, long_about = None)]
// struct Args {
//     #[clap(long, default_value = "100")]
//     sim_count: u32,
//     #[clap(long, default_value = "1.41421")]
//     explore_param_c: f32,
// }


fn main() {
    // let args = Args::parse();
    let mut encoder = simple_network::SimpleEncoder::new();
    let trainer = train::Trainer::new(&mut encoder);
    let mut player = mcts::MCTSPlayer::new_custom(100, 1.4);
    let out_path = String::from("C:/code/rl/data");
    trainer.generate_data(&mut player, 30, &out_path);
}
