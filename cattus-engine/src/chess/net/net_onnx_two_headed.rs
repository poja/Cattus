use crate::chess::chess_game::{ChessGame, ChessPosition};
use crate::chess::net::common;
use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::{NetStatistics, ValueFunction};
use crate::game::net::TwoHeadedNetBase;
use std::sync::Arc;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<ChessGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &str, cpu: bool) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, cpu, None),
        }
    }

    pub fn with_cache(model_path: &str, cpu: bool, cache: Arc<ValueFuncCache<ChessGame>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, cpu, Some(cache)),
        }
    }
}

impl ValueFunction<ChessGame> for TwoHeadedNet {
    fn evaluate(&self, position: &ChessPosition) -> (Vec<(<ChessGame as IGame>::Move, f32)>, f32) {
        self.base.evaluate(position, common::position_to_planes)
    }

    fn get_statistics(&self) -> NetStatistics {
        self.base.get_statistics()
    }
}
