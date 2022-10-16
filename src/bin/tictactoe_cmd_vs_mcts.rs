use clap::Parser;
use rl::game::common::{GamePosition, IGame};
use rl::game::mcts::MCTSPlayer;
use rl::ttt::cmd_player::TttPlayerCmd;
use rl::ttt::net::two_headed_net::TwoHeadedNet;
use rl::ttt::ttt_game::{color_to_str, TttGame};

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: String,
}

fn main() {
    const CPU: bool = true;

    let args = Args::parse();

    let value_func = Box::new(TwoHeadedNet::<CPU>::new(&args.model_path));
    let mut player1 = MCTSPlayer::new(1000, value_func);

    let mut player2 = TttPlayerCmd {};

    let mut game = TttGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
