use crate::game::common::{GameColor, GamePosition};
use crate::game::net;
use crate::game::self_play::{DataSerializer, SerializerBase};
use crate::ttt::net::common::{self, MOVES_NUM};
use crate::ttt::ttt_game::{TttGame, TttMove, TttPosition};
use itertools::Itertools;

pub struct TttSerializer {}

impl DataSerializer<TttGame> for TttSerializer {
    fn serialize_data_entry(
        &self,
        pos: TttPosition,
        probs: Vec<(TttMove, f32)>,
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
            .map(|p| p.get_raw() as u64)
            .collect_vec();

        SerializerBase::write_entry::<TttGame, MOVES_NUM>(planes, probs, winner, filename)
    }
}
