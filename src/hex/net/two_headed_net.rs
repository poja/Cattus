use crate::game::common::{GamePosition, IGame};
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;
use crate::hex::hex_game::{HexBitboard, HexGame, HexPosition, BOARD_SIZE};
use crate::hex::net::common;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase,
}

impl TwoHeadedNet {
    pub fn new(model_path: &String) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, "in_position", "out_value", "out_probs"),
        }
    }

    fn evaluate_position_impl(
        &self,
        position: &HexPosition,
    ) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        let planes = common::position_to_planes(position);
        let input = TwoHeadedNetBase::planes_to_tensor::<HexBitboard, BOARD_SIZE>(planes);
        let (val, move_scores) = self.base.run_net(input);

        let moves = position.get_legal_moves();
        let moves_probs = TwoHeadedNetBase::calc_moves_probs(moves, move_scores);

        return (val, moves_probs);
    }
}

impl ValueFunction<HexGame> for TwoHeadedNet {
    fn evaluate(&mut self, position: &HexPosition) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        let (flipped_pos, is_flipped) = common::flip_pos_if_needed(*position);
        let eval = self.evaluate_position_impl(&flipped_pos);
        return common::flip_score_if_needed(eval, is_flipped);
    }
}
