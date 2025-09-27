use crate::game::common::GameBitboard;
use crate::ttt::ttt_game::{TttBitboard, TttPosition};

pub const PLANES_NUM: usize = 3;

pub fn position_to_planes(pos: &TttPosition) -> Vec<TttBitboard> {
    let planes: [_; PLANES_NUM] = [
        // x pieces plane
        pos.pieces_x(),
        // o pieces plane
        pos.pieces_o(),
        // a plane with all ones to help NN find board edges
        TttBitboard::new_with_all(true),
    ];
    planes.to_vec()
}
