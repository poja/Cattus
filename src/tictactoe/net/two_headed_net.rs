use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;
use crate::tictactoe::net::common;
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition, TtoBitboard, BOARD_SIZE};
use std::sync::Arc;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase<TicTacToeGame>,
}

impl TwoHeadedNet {
    pub fn new(model_path: &String) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, None),
        }
    }

    pub fn with_cache(model_path: &String, cache: Arc<ValueFuncCache<TicTacToeGame>>) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, Some(cache)),
        }
    }
}

impl ValueFunction<TicTacToeGame> for TwoHeadedNet {
    fn evaluate(
        &mut self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        return self
            .base
            .evaluate::<TtoBitboard, BOARD_SIZE>(position, common::position_to_planes);
    }
}
