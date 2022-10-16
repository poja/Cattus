use crate::game::common::{GameBitboard, IGame};
use crate::hex::hex_game::{HexBitboard, HexGame, HexPosition};

pub const PLANES_NUM: usize = 3;
pub const MOVES_NUM: usize = HexGame::BOARD_SIZE * HexGame::BOARD_SIZE;

#[allow(clippy::vec_init_then_push)]
pub fn position_to_planes(pos: &HexPosition) -> Vec<HexBitboard> {
    let mut planes = Vec::new();
    /* red pieces plane */
    planes.push(pos.pieces_red());
    /* blue pieces plane */
    planes.push(pos.pieces_blue());
    /* a plane with all ones to help NN find board edges */
    planes.push(HexBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    planes
}
