use clap::Parser;
use rl::game::mcts::{MCTSPlayer, ValueFunction, ValueFunctionRand};
use rl::hex::hex_game::HexGame;
use rl::hex::net::scalar_value_net::ScalarValNet;
use rl::hex::net::two_headed_net::TwoHeadedNet;
use rl::hex::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_param_c: f32,
    #[clap(long, default_value = "_RAND_MOVES_")]
    net_type: String,
    #[clap(long, default_value = "_NONE_")]
    model_path: String,
}

fn main() {
    let args = Args::parse();

    let value_func: Box<dyn ValueFunction<HexGame>>;
    if args.net_type == "_RAND_MOVES_" {
        value_func = Box::new(ValueFunctionRand::new());
    } else if args.net_type == "scalar_net" {
        value_func = Box::new(ScalarValNet::new(&args.model_path));
    } else if args.net_type == "two_headed_net" {
        value_func = Box::new(TwoHeadedNet::new(&args.model_path));
    } else {
        panic!("unsupported net type: {}", args.net_type);
    }

    let mut player: MCTSPlayer<HexGame> =
        MCTSPlayer::new_custom(args.sim_count, args.explore_param_c, value_func);
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
