use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::self_play_cmd::{INNetworkBuilder, run_main};
use cattus::ttt::net::serializer::TttSerializer;
use cattus::ttt::net::two_headed_net::TwoHeadedNet;
use cattus::ttt::ttt_game::TttGame;
use cattus::util::Device;
use std::path::Path;
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<TttGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &Path,
        cache: Arc<ValueFuncCache<TttGame>>,
        device: Device,
        batch_size: usize,
    ) -> Box<dyn ValueFunction<TttGame>> {
        Box::new(TwoHeadedNet::with_cache(
            model_path, device, batch_size, cache,
        ))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(TttSerializer {}))
}
