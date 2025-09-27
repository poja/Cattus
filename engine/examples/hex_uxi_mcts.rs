use cattus::game::mcts::{MctsParams, MctsPlayer};
use cattus::game::model::InferenceConfig;
use cattus::hex::hex_game::HEX_STANDARD_BOARD_SIZE;
use cattus::hex::net::two_headed_net::TwoHeadedNet;
use cattus::hex::uxi;
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
    cattus::util::init_globals();

    let args = Args::parse();

    let value_func = Arc::new(TwoHeadedNet::<HEX_STANDARD_BOARD_SIZE>::new(
        &args.model_path,
        InferenceConfig::default(),
        args.batch_size,
    ));
    let player = Box::new(MctsPlayer::new(MctsParams::new(args.sim_num, value_func)));
    let mut engine = uxi::UxiEngine::new(player);
    engine.run();
}
