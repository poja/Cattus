use std::path::Path;

use crate::game::common::{GameColor, GamePosition};
use crate::game::self_play::{DataEntry, DataSerializer, SerializerBase};
use crate::hex::hex_game::HexGame;
use crate::hex::net::common::{self};
use itertools::Itertools;

pub struct HexSerializer;
impl<const BOARD_SIZE: usize> DataSerializer<HexGame<BOARD_SIZE>> for HexSerializer {
    fn serialize_data_entry(
        &self,
        entry: DataEntry<HexGame<BOARD_SIZE>>,
        filename: &Path,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(entry.winner) as i8;
        assert!(entry.pos.get_turn() == GameColor::Player1);

        #[allow(clippy::identity_op)]
        let planes = common::position_to_planes(&entry.pos)
            .into_iter()
            .flat_map(|p| {
                [
                    ((p.get_raw() >> 00) & 0xffffffffffffffff) as u64,
                    ((p.get_raw() >> 64) & 0xffffffffffffffff) as u64,
                ]
                .into_iter()
            })
            .collect_vec();

        SerializerBase::write_entry::<HexGame<BOARD_SIZE>>(planes, entry.probs, winner, filename)
    }
}
