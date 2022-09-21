use rl::game::mcts::ValueFunction;
use rl::game::players_compare_cmd::{run_main, ValueFunctionBuilder};
use rl::hex::hex_game::HexGame;
use rl::hex::net::two_headed_net::TwoHeadedNet;

struct ValueFunctionBuilderImpl {}
impl ValueFunctionBuilderImpl {
    fn new() -> Self {
        Self {}
    }
}
impl ValueFunctionBuilder<HexGame> for ValueFunctionBuilderImpl {
    fn new_value_func(&self, model_path: &String) -> Box<dyn ValueFunction<HexGame>> {
        Box::new(TwoHeadedNet::new(model_path))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(Box::new(ValueFunctionBuilderImpl::new()));
}
