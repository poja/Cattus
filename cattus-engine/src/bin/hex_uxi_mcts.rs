use cattus::game::mcts::MCTSPlayer;
use cattus::hex::hex_game::HEX_STANDARD_BOARD_SIZE;
use cattus::hex::net::two_headed_net::TwoHeadedNet;
use cattus::hex::uxi;
use cattus::utils;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long, default_value = "_NONE_")]
    model_path: String,
}

fn main() {
    utils::init_python();

    const CPU: bool = true;

    let args = Args::parse();

    let value_func = Arc::new(TwoHeadedNet::<HEX_STANDARD_BOARD_SIZE, CPU>::new(
        &args.model_path,
    ));
    let player = Box::new(MCTSPlayer::new(args.sim_num, value_func));
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
