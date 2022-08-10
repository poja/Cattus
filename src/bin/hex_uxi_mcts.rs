use clap::Parser;
use rl::game_utils::mcts::{self, ValueFunction};
use rl::hex::hex_game::HexGame;
use rl::hex::net::scalar_value_net::SimpleNetwork;
use rl::hex::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_param_c: f32,
    #[clap(long, default_value = "_RAND_MOVES_")]
    network: String,
}

fn main() {
    let args = Args::parse();

    let mut value_func_rand;
    let mut value_func_net;
    let value_func: &mut dyn ValueFunction<HexGame>;
    if args.network == "_RAND_MOVES_" {
        value_func_rand = mcts::ValueFunctionRand::new();
        value_func = &mut value_func_rand;
    } else {
        value_func_net = SimpleNetwork::new(args.network);
        value_func = &mut value_func_net;
    }

    let mut player: mcts::MCTSPlayer<HexGame> =
        mcts::MCTSPlayer::new_custom(args.sim_count, args.explore_param_c, value_func);
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
