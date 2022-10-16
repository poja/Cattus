use std::sync::Arc;

use rl::chess::chess_game::ChessGame;
use rl::chess::net::net_tf_two_headed::TwoHeadedNet;
use rl::chess::net::serializer::ChessSerializer;
use rl::game::cache::ValueFuncCache;
use rl::game::mcts::ValueFunction;
use rl::game::self_play_cmd::{run_main, INNetworkBuilder};

struct NNetworkBuilder;
impl INNetworkBuilder<ChessGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &str,
        cache: Arc<ValueFuncCache<ChessGame>>,
        cpu: bool,
    ) -> Box<dyn ValueFunction<ChessGame>> {
        if cpu {
            Box::new(TwoHeadedNet::<true>::with_cache(model_path, cache))
        } else {
            Box::new(TwoHeadedNet::<false>::with_cache(model_path, cache))
        }
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(ChessSerializer {}))
}
