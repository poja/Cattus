use itertools::Itertools;
use std::fs;

use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use crate::chess::net::common;
use crate::game::common::{GameColor, GameMove, GamePosition};
use crate::game::net;
use crate::game::self_play::DataSerializer;

pub struct ChessSerializer;
impl DataSerializer<ChessGame> for ChessSerializer {
    fn serialize_data_entry(
        &self,
        pos: ChessPosition,
        probs: Vec<(ChessMove, f32)>,
        winner: Option<GameColor>,
        filename: &str,
    ) -> std::io::Result<()> {
        /* Always serialize as turn=1 */
        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (winner, mut probs) = net::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let planes = common::position_to_planes(&pos)
            .iter()
            .map(|p| p.get_raw())
            .collect_vec();

        /* Sort moves by their indices. Important as the deserializer expect the bitmap and probs order to match */
        probs.sort_by(|(m1, _p1), (m2, _p2)| m1.to_nn_idx().cmp(&m2.to_nn_idx()));

        /* Construct moves bitmap and probs array */
        /* This is done to save disk space. Instead of saving all 1880 probabilities, we take advantage of the fact */
        /* that in chess there is no position with more than 225 legal moves (actually its probably even under 220). */
        /* A bit map of 235 bytes (1880 bits) is used to indicate which moves probabilities are actually stored in */
        /* the moves_probs array. */
        let mut moves_bitmap = [0u8; 235];
        let mut moves_probs = [-1.0f32; 225];
        for (idx, (m, prob)) in probs.into_iter().enumerate() {
            let nn_idx = m.to_nn_idx();
            let (i, j) = (nn_idx / 8, nn_idx % 8);
            moves_bitmap[i] |= 1u8 << j;
            moves_probs[idx] = prob;
        }

        let u64bytes = u64::BITS as usize / 8;
        let f32bytes = /* f32::BITS */ 32 / 8;
        let i8bytes = i8::BITS as usize / 8;
        let size = planes.len() * u64bytes + 235 + 225 * f32bytes + i8bytes;
        let mut bytes = Vec::with_capacity(size);

        /* Serialized in little indian format, should deserialized the same */
        bytes.extend(planes.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(moves_bitmap.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(moves_probs.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend((winner as i8).to_le_bytes());
        assert!(bytes.len() == size);

        /* Write to file */
        fs::write(filename, bytes)
    }
}
