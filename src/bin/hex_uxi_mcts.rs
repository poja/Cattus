use clap::Parser;
use rl::game::mcts::{MCTSPlayer, ValueFunction};
use rl::hex::hex_game::HexGame;
use rl::hex::net::two_headed_net::TwoHeadedNet;
use rl::hex::uxi;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long, default_value = "_NONE_")]
    model_path: String,
}

fn main() {
    const CPU: bool = true;

    let args = Args::parse();

    let value_func: Box<dyn ValueFunction<HexGame>> =
        Box::new(TwoHeadedNet::<CPU>::new(&args.model_path));
    let player = Box::new(MCTSPlayer::new(args.sim_num, value_func));
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
