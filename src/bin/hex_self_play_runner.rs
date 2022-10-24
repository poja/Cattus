use std::sync::Arc;

use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::self_play_cmd::{run_main, INNetworkBuilder};
use cattus::hex::hex_game::HexGame;
use cattus::hex::net::serializer::HexSerializer;
use cattus::hex::net::two_headed_net::TwoHeadedNet;

struct NNetworkBuilder;
impl INNetworkBuilder<HexGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &str,
        cache: Arc<ValueFuncCache<HexGame>>,
        cpu: bool,
    ) -> Box<dyn ValueFunction<HexGame>> {
        if cpu {
            Box::new(TwoHeadedNet::<true>::with_cache(model_path, cache))
        } else {
            Box::new(TwoHeadedNet::<false>::with_cache(model_path, cache))
        }
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(HexSerializer {}))
}
