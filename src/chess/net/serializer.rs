use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use crate::chess::net::common::{self, MOVES_NUM};
use crate::game::common::{GameColor, GamePosition};
use crate::game::net;
use crate::game::self_play::{DataSerializer, SerializerBase};
use itertools::Itertools;

pub struct ChessSerializer {}

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
        let (winner, probs) = net::flip_score_if_needed((winner, probs), is_flipped);
        assert!(pos.get_turn() == GameColor::Player1);

        let planes = common::position_to_planes(&pos)
            .iter()
            .map(|p| p.get_raw())
            .collect_vec();

        SerializerBase::write_entry::<ChessGame, MOVES_NUM>(planes, probs, winner, filename)
    }
}
