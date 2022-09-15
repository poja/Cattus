use clap::Parser;
use rl::game::mcts::{MCTSPlayer, ValueFunction};
use rl::hex::hex_game::HexGame;
use rl::hex::net::two_headed_net::TwoHeadedNet;
use rl::hex::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_param_c: f32,
    #[clap(long, default_value = "_NONE_")]
    model_path: String,
}

fn main() {
    let args = Args::parse();

    let value_func: Box<dyn ValueFunction<HexGame>> = Box::new(TwoHeadedNet::new(&args.model_path));
    let mut player: MCTSPlayer<HexGame> =
        MCTSPlayer::new_custom(args.sim_count, args.explore_param_c, value_func);
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
