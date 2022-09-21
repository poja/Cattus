use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use crate::chess::net::common::{self, MOVES_NUM};
use crate::game::common::{GameColor, GameMove, GamePosition};
use crate::game::self_play::DataSerializer;
use itertools::Itertools;
use std::fs;

pub struct ChessSerializer {}

impl ChessSerializer {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataSerializer<ChessGame> for ChessSerializer {
    fn serialize_data_entry_to_file(
        &self,
        pos: ChessPosition,
        probs: Vec<(ChessMove, f32)>,
        winner: Option<GameColor>,
        filename: String,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = common::flip_pos_if_needed(pos);
        let (winner, probs) = common::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let planes = common::position_to_planes(&pos)
            .iter()
            .map(|p| p.get_raw())
            .collect_vec();

        let mut probs_vec = vec![0.0; MOVES_NUM];
        for (m, prob) in probs {
            probs_vec[m.to_nn_idx()] = prob;
        }

        let json_obj = json::object! {
            planes: planes,
            probs: probs_vec,
            winner: winner
        };

        let json_str = json_obj.dump();
        fs::write(filename, json_str)?;

        return Ok(());
    }
}
