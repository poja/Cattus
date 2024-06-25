use cattus::chess::chess_game::ChessGame;
use cattus::chess::net::net_onnx_two_headed::TwoHeadedNet;
use cattus::chess::net::serializer::ChessSerializer;
use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::game::self_play_cmd::{run_main, INNetworkBuilder};
use std::sync::Arc;

struct NNetworkBuilder;
impl INNetworkBuilder<ChessGame> for NNetworkBuilder {
    fn build_net(
        &self,
        model_path: &str,
        cache: Arc<ValueFuncCache<ChessGame>>,
        cpu: bool,
    ) -> Box<dyn ValueFunction<ChessGame>> {
        Box::new(TwoHeadedNet::with_cache(model_path, cpu, cache))
    }
}

fn main() -> std::io::Result<()> {
    run_main(Box::new(NNetworkBuilder {}), Box::new(ChessSerializer {}))
}
