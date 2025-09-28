use cattus::chess::ChessGame;
use cattus::mcts::cache::ValueFuncCache;
use cattus::mcts::value_func::ValueFunction;
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use cattus_self_play::self_play_cmd::{run_main, INNetworkBuilder};
use cattus_self_play::serialize::chess::ChessSerializer;
use std::path::Path;
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<ChessGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &Path,
        inference_cfg: InferenceConfig,
        batch_size: usize,
        cache: Arc<ValueFuncCache<ChessGame>>,
    ) -> Box<dyn ValueFunction<ChessGame>> {
        Box::new(NNetwork::new(model_path, inference_cfg, batch_size, Some(cache)))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder), Box::new(ChessSerializer))
}
