use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::model::InferenceConfig;
use cattus::ttt::net::two_headed_net::TwoHeadedNet;
use cattus::ttt::ttt_game::TttGame;
use cattus_self_play::self_play_cmd::{run_main, INNetworkBuilder};
use cattus_self_play::serialize::ttt::TttSerializer;
use std::path::Path;
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<TttGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &Path,
        inference_cfg: InferenceConfig,
        batch_size: usize,
        cache: Arc<ValueFuncCache<TttGame>>,
    ) -> Box<dyn ValueFunction<TttGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, inference_cfg, batch_size, cache))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder), Box::new(TttSerializer))
}
