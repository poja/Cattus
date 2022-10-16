use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;
use crate::hex::hex_game::{HexGame, HexPosition};
use crate::hex::net::common;
use std::sync::Arc;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<HexGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &str) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, None),
        }
    }

    pub fn with_cache(model_path: &str, cache: Arc<ValueFuncCache<HexGame>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, Some(cache)),
        }
    }
}

impl ValueFunction<HexGame> for TwoHeadedNet {
    fn evaluate(&mut self, position: &HexPosition) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        self.base.evaluate(position, common::position_to_planes)
    }
}
