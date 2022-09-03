use clap::Parser;
use rl::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition, color_to_str};
use rl::game::common::{GamePosition, IGame};
use rl::tictactoe::cmd_player::TttPlayerCmd;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::game::mcts::MCTSPlayer;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model_path: String,
}

fn main() {
    let args = Args::parse();
    let position = TicTacToePosition::new();
    let mut player2 = TttPlayerCmd::new();

    let mut value_func = TwoHeadedNet::new(args.model_path);
    let mut mcts_player = MCTSPlayer::new_custom(1000, 1.41421, &mut value_func);

    let (final_pos, winner) = TicTacToeGame::play_until_over(&position, &mut mcts_player, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();

}
