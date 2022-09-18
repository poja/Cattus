use crate::game::common::{GamePosition, IGame};
use crate::game::mcts::ValueFunction;
use crate::game::net::TwoHeadedNetBase;
use crate::tictactoe::net::common;
use crate::tictactoe::net::encoder::Encoder;
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition};
use itertools::Itertools;

pub struct TwoHeadedNet {
    base: TwoHeadedNetBase,
    encoder: Encoder,
}

impl TwoHeadedNet {
    pub fn new(model_path: &String) -> Self {
        Self {
            base: TwoHeadedNetBase::new(model_path, "in_position", "out_value", "out_probs"),
            encoder: Encoder::new(),
        }
    }

    fn evaluate_position_impl(
        &self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        let input = self.encoder.encode_position(position);
        let (val, probs) = self.base.run_net(input);

        let moves = position.get_legal_moves();
        let moves_probs = moves
            .iter()
            .map(|move_| ((*move_), probs[move_.to_idx() as usize]))
            .collect_vec();

        return (val, moves_probs);
    }
}

impl ValueFunction<TicTacToeGame> for TwoHeadedNet {
    fn evaluate(
        &mut self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        let (flipped_pos, is_flipped) = common::flip_pos_if_needed(*position);
        let eval = self.evaluate_position_impl(&flipped_pos);
        return common::flip_score_if_needed(eval, is_flipped);
    }
}
