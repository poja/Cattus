use crate::game::common::{GameColor, GamePosition};
use crate::game::net;
use crate::game::self_play::{DataSerializer, SerializerBase};
use crate::tictactoe::net::common::{self, MOVES_NUM};
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToeMove, TicTacToePosition};
use itertools::Itertools;

pub struct TicTacToeSerializer {}

impl TicTacToeSerializer {
    pub fn new() -> Self {
        Self {}
    }
}

impl DataSerializer<TicTacToeGame> for TicTacToeSerializer {
    fn serialize_data_entry(
        &self,
        pos: TicTacToePosition,
        probs: Vec<(TicTacToeMove, f32)>,
        winner: Option<GameColor>,
        filename: &String,
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

        return SerializerBase::write_entry::<TicTacToeGame, MOVES_NUM>(
            planes, probs, winner, filename,
        );
    }
}
