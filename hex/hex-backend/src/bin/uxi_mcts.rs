use clap::Parser;
use hex_backend::mcts;
use hex_backend::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_count: u32,
}

fn main() {
    let args = Args::parse();
    let mut player = mcts::MCTSPlayer::with_simulations_per_move(args.sim_count);
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
