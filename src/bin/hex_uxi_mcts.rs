use clap::Parser;
use cattus::game::mcts::{MCTSPlayer, ValueFunction};
use cattus::hex::hex_game::HexGame;
use cattus::hex::net::two_headed_net::TwoHeadedNet;
use cattus::hex::uxi;

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
