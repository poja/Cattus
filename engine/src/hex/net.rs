use crate::game::Bitboard;
use crate::hex::{HexBitboard, HexGame, HexMove, HexPosition};
use crate::mcts::value_func::ValueFunction;
use crate::net::NNetwork;

impl<const BOARD_SIZE: usize> ValueFunction<HexGame<BOARD_SIZE>> for NNetwork<HexGame<BOARD_SIZE>> {
    fn evaluate(&self, position: &HexPosition<BOARD_SIZE>) -> (Vec<(HexMove<BOARD_SIZE>, f32)>, f32) {
        self.evaluate(position, position_to_planes)
    }
}

pub const PLANES_NUM: usize = 3;

pub fn position_to_planes<const BOARD_SIZE: usize>(pos: &HexPosition<BOARD_SIZE>) -> Vec<HexBitboard<BOARD_SIZE>> {
    let planes: [_; PLANES_NUM] = [
        /* red pieces plane */
        pos.pieces_red(),
        /* blue pieces plane */
        pos.pieces_blue(),
        /* a plane with all ones to help NN find board edges */
        HexBitboard::full(true),
    ];
    planes.to_vec()
}
