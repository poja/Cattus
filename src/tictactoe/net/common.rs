use crate::game::common::Bitboard;
use crate::tictactoe::tictactoe_game::{TicTacToePosition, TtoBitboard, BOARD_SIZE};

pub const PLANES_NUM: usize = 3;
pub const MOVES_NUM: usize = BOARD_SIZE * BOARD_SIZE;

pub fn position_to_planes(pos: &TicTacToePosition) -> Vec<TtoBitboard> {
    let mut planes = Vec::new();
    /* x pieces plane */
    planes.push(pos.pieces_x());
    /* o pieces plane */
    planes.push(pos.pieces_o());
    /* a plane with all ones to help NN find board edges */
    planes.push(TtoBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    return planes;
}
