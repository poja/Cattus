use crate::chess::chess_game::{ChessGame, ChessPosition};
use crate::chess::net::common;
use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::model::InferenceConfig;
use crate::game::net::TwoHeadedNetBase;
use std::path::Path;
use std::sync::Arc;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<ChessGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &Path, inference_cfg: InferenceConfig, batch_size: usize) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, inference_cfg, batch_size, None),
        }
    }

    pub fn with_cache(
        model_path: &Path,
        inference_cfg: InferenceConfig,
        batch_size: usize,
        cache: Arc<ValueFuncCache<ChessGame>>,
    ) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, inference_cfg, batch_size, Some(cache)),
        }
    }
}

impl ValueFunction<ChessGame> for TwoHeadedNet {
    fn evaluate(&self, position: &ChessPosition) -> (Vec<(<ChessGame as IGame>::Move, f32)>, f32) {
        self.base.evaluate(position, common::position_to_planes)
    }
}
