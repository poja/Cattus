use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::{NetStatistics, ValueFunction};
use crate::game::net::TwoHeadedNetBase;
use crate::ttt::net::common;
use crate::ttt::ttt_game::{TttGame, TttPosition};
use crate::utils::Device;
use std::sync::Arc;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<TttGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &str, device: Device) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, device, None),
        }
    }

    pub fn with_cache(
        model_path: &str,
        device: Device,
        cache: Arc<ValueFuncCache<TttGame>>,
    ) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, device, Some(cache)),
        }
    }
}

impl ValueFunction<TttGame> for TwoHeadedNet {
    fn evaluate(&self, position: &TttPosition) -> (Vec<(<TttGame as IGame>::Move, f32)>, f32) {
        self.base.evaluate(position, common::position_to_planes)
    }

    fn get_statistics(&self) -> NetStatistics {
        self.base.get_statistics()
    }
}
