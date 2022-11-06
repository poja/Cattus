use crate::game::common::GameBitboard;
use crate::ttt::ttt_game::{TttBitboard, TttPosition};

pub const PLANES_NUM: usize = 3;

#[allow(clippy::vec_init_then_push)]
pub fn position_to_planes(pos: &TttPosition) -> Vec<TttBitboard> {
    let mut planes = Vec::new();
    /* x pieces plane */
    planes.push(pos.pieces_x());
    /* o pieces plane */
    planes.push(pos.pieces_o());
    /* a plane with all ones to help NN find board edges */
    planes.push(TttBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    planes
}
