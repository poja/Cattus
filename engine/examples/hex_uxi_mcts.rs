use cattus::game::mcts::MCTSPlayer;
use cattus::hex::hex_game::HEX_STANDARD_BOARD_SIZE;
use cattus::hex::net::two_headed_net::TwoHeadedNet;
use cattus::hex::uxi;
use cattus::util::Device;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: usize,
    #[clap(long)]
    model_path: PathBuf,
}

fn main() {
    cattus::util::init_globals(None);

    let args = Args::parse();

    let value_func = Arc::new(TwoHeadedNet::<HEX_STANDARD_BOARD_SIZE>::new(
        &args.model_path,
        args.batch_size,
        Device::Cpu,
    ));
    let player = Box::new(MCTSPlayer::new(args.sim_num, value_func));
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
