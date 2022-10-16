use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;
use crate::ttt::net::common;
use crate::ttt::ttt_game::{TttGame, TttPosition};
use std::sync::Arc;

pub struct TwoHeadedNet<const CPU: bool> {
    base: TwoHeadedNetBase<TttGame, CPU>,
}

impl<const CPU: bool> TwoHeadedNet<CPU> {
    pub fn new(model_path: &str) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, None),
        }
    }

    pub fn with_cache(model_path: &str, cache: Arc<ValueFuncCache<TttGame>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, Some(cache)),
        }
    }
}

impl<const CPU: bool> ValueFunction<TttGame> for TwoHeadedNet<CPU> {
    fn evaluate(&mut self, position: &TttPosition) -> (f32, Vec<(<TttGame as IGame>::Move, f32)>) {
        self.base.evaluate(position, common::position_to_planes)
    }
}
