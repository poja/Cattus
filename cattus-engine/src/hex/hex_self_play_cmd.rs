use crate::game::cache::ValueFuncCache;
use crate::game::mcts::ValueFunction;
use crate::game::self_play_cmd::{self, INNetworkBuilder};
use crate::hex::hex_game::HexGame;
use crate::hex::net::serializer::HexSerializer;
use crate::hex::net::two_headed_net::TwoHeadedNet;
use crate::utils::Device;
use std::sync::Arc;

struct NNetworkBuilder<const BOARD_SIZE: usize>;
impl<const BOARD_SIZE: usize> INNetworkBuilder<HexGame<BOARD_SIZE>>
    for NNetworkBuilder<BOARD_SIZE>
{
    fn build_net(
        &self,
        model_path: &str,
        cache: Arc<ValueFuncCache<HexGame<BOARD_SIZE>>>,
        device: Device,
    ) -> Box<dyn ValueFunction<HexGame<BOARD_SIZE>>> {
        Box::new(TwoHeadedNet::<BOARD_SIZE>::with_cache(
            model_path, device, cache,
        ))
    }
}

pub fn run_main<const BOARD_SIZE: usize>() -> std::io::Result<()> {
    self_play_cmd::run_main(
        Box::new(NNetworkBuilder::<BOARD_SIZE> {}),
        Box::new(HexSerializer {}),
    )
}
