use clap::Parser;
use rl::game::common::{GamePosition, IGame};
use rl::game::mcts::MCTSPlayer;
use rl::tictactoe::cmd_player::TttPlayerCmd;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::tictactoe::tictactoe_game::{color_to_str, TicTacToeGame};

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: String,
}

fn main() {
    let args = Args::parse();

    let value_func = Box::new(TwoHeadedNet::new(&args.model_path));
    let mut player1 = MCTSPlayer::new_custom(1000, 1.41421, value_func);

    let mut player2 = TttPlayerCmd::new();

    let mut game = TicTacToeGame::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();
}
