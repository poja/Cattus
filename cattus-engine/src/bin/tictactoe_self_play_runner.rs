use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::self_play_cmd::{run_main, INNetworkBuilder};
use cattus::ttt::net::serializer::TttSerializer;
use cattus::ttt::net::two_headed_net::TwoHeadedNet;
use cattus::ttt::ttt_game::TttGame;
use cattus::utils::Device;
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<TttGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &str,
        cache: Arc<ValueFuncCache<TttGame>>,
        device: Device,
    ) -> Box<dyn ValueFunction<TttGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, device, cache))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(TttSerializer {}))
}
