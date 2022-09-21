use std::fs;

use itertools::Itertools;

use crate::game::common::{GameColor, GamePosition};
use crate::game::self_play::DataSerializer;
use crate::hex::hex_game::{HexGame, HexMove, HexPosition, BOARD_SIZE};
use crate::hex::net::common;

pub struct HexSerializer {}

impl HexSerializer {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataSerializer<HexGame> for HexSerializer {
    fn serialize_data_entry_to_file(
        &self,
        pos: HexPosition,
        probs: Vec<(HexMove, f32)>,
        winner: Option<GameColor>,
        filename: String,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = common::flip_pos_if_needed(pos);
        let (winner, probs) = common::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let planes = common::position_to_planes(&pos)
            .into_iter()
            .flat_map(|p| {
                [
                    /* TODO !!! possible little/big indian bug */
                    /* Need to use some library (protobuf) to ensure writes and reads are done in the same way */
                    ((p.get_raw() >> 00) & 0xffffffffffffffff) as u64,
                    ((p.get_raw() >> 64) & 0xffffffffffffffff) as u64,
                ]
                .into_iter()
            })
            .into_iter()
            .collect_vec();

        let mut probs_vec = vec![0.0; BOARD_SIZE * BOARD_SIZE];
        for (m, prob) in probs {
            probs_vec[m.to_idx()] = prob;
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
