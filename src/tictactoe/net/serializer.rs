use std::fs;

use crate::game::common::GameColor;
use crate::game::common::GamePosition;
use crate::game::self_play::DataSerializer;
use crate::tictactoe::net;
use crate::tictactoe::tictactoe_game::{
    TicTacToeGame, TicTacToeMove, TicTacToePosition, BOARD_SIZE,
};

pub struct TicTacToeSerializer {}

impl TicTacToeSerializer {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataSerializer<TicTacToeGame> for TicTacToeSerializer {
    fn serialize_data_entry_to_file(
        &self,
        pos: TicTacToePosition,
        probs: Vec<(TicTacToeMove, f32)>,
        winner: Option<GameColor>,
        filename: String,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::common::flip_pos_if_needed(pos);
        let (winner, probs) = net::common::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let mut pos_vec = Vec::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                pos_vec.push(GameColor::to_idx(pos.get_tile(r, c)) as f32);
            }
        }

        let mut probs_vec = vec![0.0; (BOARD_SIZE * BOARD_SIZE) as usize];
        for (m, prob) in probs {
            probs_vec[m.to_idx() as usize] = prob;
        }

        let json_obj = json::object! {
            position: pos_vec,
            moves_probabilities: probs_vec,
            winner: winner
        };

        let json_str = json_obj.dump();
        fs::write(filename, json_str)?;

        return Ok(());
    }
}
