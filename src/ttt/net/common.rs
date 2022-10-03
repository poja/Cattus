use crate::game::common::{GameBitboard, IGame};
use crate::ttt::ttt_game::{TttBitboard, TttGame, TttPosition};

pub const PLANES_NUM: usize = 3;
pub const MOVES_NUM: usize = TttGame::BOARD_SIZE * TttGame::BOARD_SIZE;

pub fn position_to_planes(pos: &TttPosition) -> Vec<TttBitboard> {
    let mut planes = Vec::new();
    /* x pieces plane */
    planes.push(pos.pieces_x());
    /* o pieces plane */
    planes.push(pos.pieces_o());
    /* a plane with all ones to help NN find board edges */
    planes.push(TttBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    return planes;
}
