use crate::game::cache::ValueFuncCache;
use crate::game::mcts::{NetStatistics, ValueFunction};
use crate::game::net::TwoHeadedNetBase;
use crate::hex::hex_game::{HexGame, HexMove, HexPosition};
use crate::hex::net::common;
use crate::utils::Device;
use std::sync::Arc;

pub struct TwoHeadedNet<const BOARD_SIZE: usize> {
    base: TwoHeadedNetBase<HexGame<BOARD_SIZE>>,
}

impl<const BOARD_SIZE: usize> TwoHeadedNet<BOARD_SIZE> {
    pub fn new(model_path: &str, device: Device) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, device, None),
        }
    }

    pub fn with_cache(
        model_path: &str,
        device: Device,
        cache: Arc<ValueFuncCache<HexGame<BOARD_SIZE>>>,
    ) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, device, Some(cache)),
        }
    }
}

impl<const BOARD_SIZE: usize> ValueFunction<HexGame<BOARD_SIZE>> for TwoHeadedNet<BOARD_SIZE> {
    fn evaluate(
        &self,
        position: &HexPosition<BOARD_SIZE>,
    ) -> (Vec<(HexMove<BOARD_SIZE>, f32)>, f32) {
        self.base.evaluate(position, common::position_to_planes)
    }

    fn get_statistics(&self) -> NetStatistics {
        self.base.get_statistics()
    }
}
