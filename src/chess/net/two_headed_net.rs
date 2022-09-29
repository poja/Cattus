use crate::chess::chess_game::{ChessBitboard, ChessGame, ChessPosition, BOARD_SIZE};
use crate::chess::net::common;
use crate::game::common::IGame;
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase,
}

impl TwoHeadedNet {
    pub fn new(model_path: &String) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path),
        }
    }
}

impl ValueFunction<ChessGame> for TwoHeadedNet {
    fn evaluate(
        &mut self,
        position: &ChessPosition,
    ) -> (f32, Vec<(<ChessGame as IGame>::Move, f32)>) {
        return self.base.evaluate::<ChessGame, ChessBitboard, BOARD_SIZE>(
            position,
            common::position_to_planes,
        );
    }
}
