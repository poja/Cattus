use cattus::chess::chess_game::ChessGame;
use cattus::chess::net::net_onnx_two_headed::TwoHeadedNet;
use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::util::Device;
use cattus_self_play::self_play_cmd::{INNetworkBuilder, run_main};
use cattus_self_play::serialize::chess::ChessSerializer;
use std::path::Path;
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<ChessGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &Path,
        cache: Arc<ValueFuncCache<ChessGame>>,
        device: Device,
        batch_size: usize,
    ) -> Box<dyn ValueFunction<ChessGame>> {
        Box::new(TwoHeadedNet::with_cache(
            model_path, device, batch_size, cache,
        ))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(ChessSerializer {}))
}
