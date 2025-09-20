use crate::self_play_cmd::{self, INNetworkBuilder};
use crate::serialize::hex::HexSerializer;
use cattus::game::cache::ValueFuncCache;
use cattus::game::mcts::ValueFunction;
use cattus::hex::hex_game::HexGame;
use cattus::hex::net::two_headed_net::TwoHeadedNet;
use cattus::util::Device;
use std::path::Path;
use std::sync::Arc;

struct NNetworkBuilder<const BOARD_SIZE: usize>;
impl<const BOARD_SIZE: usize> INNetworkBuilder<HexGame<BOARD_SIZE>>
    for NNetworkBuilder<BOARD_SIZE>
{
    fn build_net(
        &self,
        model_path: &Path,
        cache: Arc<ValueFuncCache<HexGame<BOARD_SIZE>>>,
        device: Device,
        batch_size: usize,
    ) -> Box<dyn ValueFunction<HexGame<BOARD_SIZE>>> {
        Box::new(TwoHeadedNet::<BOARD_SIZE>::with_cache(
            model_path, device, batch_size, cache,
        ))
    }
}

pub fn run_main<const BOARD_SIZE: usize>() -> std::io::Result<()> {
    self_play_cmd::run_main(
        Box::new(NNetworkBuilder::<BOARD_SIZE> {}),
        Box::new(HexSerializer {}),
    )
}
