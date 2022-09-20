use crate::game::common::{Bitboard, GameColor, GamePosition, IGame};
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition, TtoBitboard};

pub const PLANES_NUM: usize = 3;

pub fn flip_pos_if_needed(pos: TicTacToePosition) -> (TicTacToePosition, bool) {
    if pos.get_turn() == GameColor::Player1 {
        return (pos, false);
    } else {
        return (TicTacToePosition::flip_of(&pos), true);
    }
}

pub fn flip_score_if_needed(
    net_res: (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>),
    pos_flipped: bool,
) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
    if !pos_flipped {
        return net_res;
    } else {
        let (val, moves_probs) = net_res;
        let val = -val;
        return (val, moves_probs);
    }
}

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
