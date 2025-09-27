use cattus::chess::chess_game::ChessGame;
use cattus::chess::net::net_two_headed::TwoHeadedNet;
use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::model::InferenceConfig;
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
        Box::new(TwoHeadedNet::with_cache(model_path, inference_cfg, batch_size, cache))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder), Box::new(ChessSerializer))
}
