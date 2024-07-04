use cattus::game::common::{GamePosition, IGame};
use cattus::game::mcts::MCTSPlayer;
use cattus::ttt::cmd_player::TttPlayerCmd;
use cattus::ttt::net::two_headed_net::TwoHeadedNet;
use cattus::ttt::ttt_game::{color_to_str, TttGame};
use cattus::utils::{self, Device};
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: String,
}

fn main() {
    utils::init_globals(None);

    let args = Args::parse();

    let value_func = Arc::new(TwoHeadedNet::new(&args.model_path, Device::Cpu));
    let mut player1 = MCTSPlayer::new(1000, value_func);

    let mut player2 = TttPlayerCmd {};

    let mut game = TttGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
