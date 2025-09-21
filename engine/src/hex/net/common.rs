use crate::game::common::GameBitboard;
use crate::hex::hex_game::{HexBitboard, HexPosition};

pub const PLANES_NUM: usize = 3;

#[allow(clippy::vec_init_then_push)]
pub fn position_to_planes<const BOARD_SIZE: usize>(
    pos: &HexPosition<BOARD_SIZE>,
) -> Vec<HexBitboard<BOARD_SIZE>> {
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
