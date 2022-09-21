use rl::chess::chess_game::ChessGame;
use rl::chess::net::two_headed_net::TwoHeadedNet;
use rl::game::mcts::ValueFunction;
use rl::game::players_compare_cmd::{run_main, ValueFunctionBuilder};

struct ValueFunctionBuilderImpl {}
impl ValueFunctionBuilderImpl {
    fn new() -> Self {
        Self {}
    }
}
impl ValueFunctionBuilder<ChessGame> for ValueFunctionBuilderImpl {
    fn new_value_func(&self, model_path: &String) -> Box<dyn ValueFunction<ChessGame>> {
        Box::new(TwoHeadedNet::new(model_path))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(Box::new(ValueFunctionBuilderImpl::new()));
}
