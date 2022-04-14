use clap::Parser;
use rl::game_utils::mcts;
use rl::hex::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_param_c: f32,
}

fn main() {
    let args = Args::parse();
    let mut player = mcts::MCTSPlayer::new_custom(args.sim_count, args.explore_param_c);
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
