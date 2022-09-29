use crate::game::common::Bitboard;
use crate::hex::hex_game::{HexBitboard, HexPosition, BOARD_SIZE};

pub const PLANES_NUM: usize = 3;
pub const MOVES_NUM: usize = BOARD_SIZE * BOARD_SIZE;

pub fn position_to_planes(pos: &HexPosition) -> Vec<HexBitboard> {
    let mut planes = Vec::new();
    /* red pieces plane */
    planes.push(pos.pieces_red());
    /* blue pieces plane */
    planes.push(pos.pieces_blue());
    /* a plane with all ones to help NN find board edges */
    planes.push(HexBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    return planes;
}
