use std::sync::Arc;

use rl::game::cache::ValueFuncCache;
use rl::game::mcts::ValueFunction;
use rl::game::self_play_cmd::{run_main, INNetworkBuilder};
use rl::hex::hex_game::HexGame;
use rl::hex::net::serializer::HexSerializer;
use rl::hex::net::two_headed_net::TwoHeadedNet;

struct NNetworkBuilder;
impl INNetworkBuilder<HexGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &String,
        cache: Arc<ValueFuncCache<HexGame>>,
    ) -> Box<dyn ValueFunction<HexGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, cache))
    }
}

fn main() -> std::io::Result<()> {
    return run_main(Box::new(NNetworkBuilder {}), Box::new(HexSerializer::new()));
}
