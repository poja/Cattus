use std::sync::Arc;

use crate::chess::chess_game::{ChessGame, ChessPosition};
use crate::chess::net::common;
use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::{NetStatistics, ValueFunction};
use crate::game::net::TwoHeadedNetBase;

pub struct TwoHeadedNet<const CPU: bool> {
    base: TwoHeadedNetBase<ChessGame, CPU>,
}

impl<const CPU: bool> TwoHeadedNet<CPU> {
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

impl<const CPU: bool> ValueFunction<ChessGame> for TwoHeadedNet<CPU> {
    fn evaluate(&self, position: &ChessPosition) -> (f32, Vec<(<ChessGame as IGame>::Move, f32)>) {
        self.base.evaluate(position, common::position_to_planes)
    }

    fn get_statistics(&self) -> NetStatistics {
        self.base.get_statistics()
    }
}
