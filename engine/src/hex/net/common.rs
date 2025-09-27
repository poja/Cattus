use crate::game::common::GameBitboard;
use crate::hex::hex_game::{HexBitboard, HexPosition};

pub const PLANES_NUM: usize = 3;

pub fn position_to_planes<const BOARD_SIZE: usize>(pos: &HexPosition<BOARD_SIZE>) -> Vec<HexBitboard<BOARD_SIZE>> {
    let planes: [_; PLANES_NUM] = [
        /* red pieces plane */
        pos.pieces_red(),
        /* blue pieces plane */
        pos.pieces_blue(),
        /* a plane with all ones to help NN find board edges */
        HexBitboard::new_with_all(true),
    ];
    planes.to_vec()
}
