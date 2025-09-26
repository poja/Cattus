use cattus::game::common::{GamePosition, IGame};
use cattus::game::mcts::{MctsParams, MctsPlayer};
use cattus::ttt::cmd_player::TttPlayerCmd;
use cattus::ttt::net::two_headed_net::TwoHeadedNet;
use cattus::ttt::ttt_game::{TttGame, color_to_str};
use cattus::util::Device;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: PathBuf,
    #[clap(long)]
    batch_size: usize,
}

fn main() {
    cattus::util::init_globals(None);

    let args = Args::parse();

    let value_func = Arc::new(TwoHeadedNet::new(
        &args.model_path,
        args.batch_size,
        Device::Cpu,
    ));
    let mut player1 = MctsPlayer::new(MctsParams::new(1000, value_func));

    let mut player2 = TttPlayerCmd;

    let mut game = TttGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
