use rl::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition, color_to_str};
use rl::game::common::{GamePosition, IGame};
use rl::tictactoe::cmd_player::TttPlayerCmd;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::game::mcts::MCTSPlayer;

fn main() {
    let position = TicTacToePosition::new();
    let mut player2 = TttPlayerCmd::new();

    let mut value_func = TwoHeadedNet::new("/Users/yishai/work/RL/workarea/models/model_220827_220925".to_string());
    let mut mcts_player = MCTSPlayer::new_custom(1000, 1.41421, &mut value_func);

    let (final_pos, winner) = TicTacToeGame::play_until_over(&position, &mut mcts_player, &mut player2);
    println!("The winner is: {}, details below:", color_to_str(winner));
    final_pos.print();

}
