use rl::game::mcts::ValueFunction;
use rl::game::players_compare_cmd::{run_main, ValueFunctionBuilder};
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::tictactoe::tictactoe_game::TicTacToeGame;

struct ValueFunctionBuilderImpl {}
impl ValueFunctionBuilderImpl {
    fn new() -> Self {
        Self {}
    }
}
impl ValueFunctionBuilder<TicTacToeGame> for ValueFunctionBuilderImpl {
    fn new_value_func(&self, model_path: &String) -> Box<dyn ValueFunction<TicTacToeGame>> {
        Box::new(TwoHeadedNet::new(model_path))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(Box::new(ValueFunctionBuilderImpl::new()));
}
