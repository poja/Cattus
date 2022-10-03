use std::sync::Arc;

use rl::game::cache::ValueFuncCache;
use rl::game::mcts::ValueFunction;
use rl::game::self_play_cmd::{run_main, INNetworkBuilder};
use rl::ttt::net::serializer::TttSerializer;
use rl::ttt::net::two_headed_net::TwoHeadedNet;
use rl::ttt::ttt_game::TttGame;

struct NNetworkBuilder;
impl INNetworkBuilder<TttGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &String,
        cache: Arc<ValueFuncCache<TttGame>>,
    ) -> Box<dyn ValueFunction<TttGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, cache))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(Box::new(NNetworkBuilder {}), Box::new(TttSerializer::new()));
}
