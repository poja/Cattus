use std::sync::Arc;

use crate::game::cache::ValueFuncCache;
use crate::game::mcts::{NetStatistics, ValueFunction};
use crate::game::net::TwoHeadedNetBase;
use crate::hex::hex_game::{HexGame, HexMove, HexPosition};
use crate::hex::net::common;

pub struct TwoHeadedNet<const BOARD_SIZE: usize, const CPU: bool> {
    base: TwoHeadedNetBase<HexGame<BOARD_SIZE>, CPU>,
}

impl<const BOARD_SIZE: usize, const CPU: bool> TwoHeadedNet<BOARD_SIZE, CPU> {
    pub fn new(model_path: &str) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, None),
        }
    }

    pub fn with_cache(model_path: &str, cache: Arc<ValueFuncCache<HexGame<BOARD_SIZE>>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, Some(cache)),
        }
    }
}

impl<const BOARD_SIZE: usize, const CPU: bool> ValueFunction<HexGame<BOARD_SIZE>>
    for TwoHeadedNet<BOARD_SIZE, CPU>
{
    fn evaluate(
        &self,
        position: &HexPosition<BOARD_SIZE>,
    ) -> (f32, Vec<(HexMove<BOARD_SIZE>, f32)>) {
        self.base.evaluate(position, common::position_to_planes)
    }

    fn get_statistics(&self) -> NetStatistics {
        self.base.get_statistics()
    }
}
