use crate::game::common::{GameColor, GamePosition};
use crate::game::net;
use crate::game::self_play::{DataSerializer, SerializerBase};
use crate::hex::hex_game::{HexGame, HexMove, HexPosition};
use crate::hex::net::common::{self, MOVES_NUM};
use itertools::Itertools;

pub struct HexSerializer {}

impl HexSerializer {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataSerializer<HexGame> for HexSerializer {
    fn serialize_data_entry(
        &self,
        pos: HexPosition,
        probs: Vec<(HexMove, f32)>,
        winner: Option<GameColor>,
        filename: &String,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (winner, probs) = net::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let planes = common::position_to_planes(&pos)
            .into_iter()
            .flat_map(|p| {
                [
                    ((p.get_raw() >> 00) & 0xffffffffffffffff) as u64,
                    ((p.get_raw() >> 64) & 0xffffffffffffffff) as u64,
                ]
                .into_iter()
            })
            .into_iter()
            .collect_vec();

        return SerializerBase::write_entry::<HexGame, MOVES_NUM>(planes, probs, winner, filename);
    }
}
