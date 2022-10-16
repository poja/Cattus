use std::sync::Arc;

use crate::chess::chess_game::{ChessGame, ChessPosition};
use crate::chess::net::common;
use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<ChessGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &str) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, None),
        }
    }

    pub fn with_cache(model_path: &str, cache: Arc<ValueFuncCache<ChessGame>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, Some(cache)),
        }
    }
}

impl ValueFunction<ChessGame> for TwoHeadedNet {
    fn evaluate(
        &mut self,
        position: &ChessPosition,
    ) -> (f32, Vec<(<ChessGame as IGame>::Move, f32)>) {
        self.base.evaluate(position, common::position_to_planes)
    }
}
