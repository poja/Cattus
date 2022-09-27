use std::sync::Arc;

use rl::game::cache::ValueFuncCache;
use rl::game::mcts::ValueFunction;
use rl::game::self_play_cmd::{run_main, INNetworkBuilder};
use rl::tictactoe::net::serializer::TicTacToeSerializer;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::tictactoe::tictactoe_game::TicTacToeGame;

struct NNetworkBuilder;
impl INNetworkBuilder<TicTacToeGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &String,
        cache: Arc<ValueFuncCache<TicTacToeGame>>,
    ) -> Box<dyn ValueFunction<TicTacToeGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, cache))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(
        Box::new(NNetworkBuilder {}),
        Box::new(TicTacToeSerializer::new()),
    );
}
