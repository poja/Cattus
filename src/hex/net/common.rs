use crate::game::common::{GameColor, GamePosition, IGame};
use crate::hex::hex_game::{Bitboard, HexGame, HexMove, HexPosition};
use itertools::Itertools;

pub const PLANES_NUM: usize = 3;

pub fn flip_pos_if_needed(pos: HexPosition) -> (HexPosition, bool) {
    if pos.get_turn() == GameColor::Player1 {
        return (pos, false);
    } else {
        return (HexPosition::flip_of(&pos), true);
    }
}

pub fn flip_score_if_needed(
    net_res: (f32, Vec<(<HexGame as IGame>::Move, f32)>),
    pos_flipped: bool,
) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
    if !pos_flipped {
        return net_res;
    } else {
        let (val, moves_probs) = net_res;

        /* Flip scalar value */
        let val = -val;

        /* Flip moves */
        let moves_probs = moves_probs
            .iter()
            .map(|(m, p)| (HexMove::new(m.column(), m.row()), *p))
            .collect_vec();

        return (val, moves_probs);
    }
}

pub fn position_to_planes(pos: &HexPosition) -> Vec<Bitboard> {
    let mut planes = Vec::new();
    /* red pieces plane */
    planes.push(pos.pieces_red());
    /* blue pieces plane */
    planes.push(pos.pieces_blue());
    /* a plane with all ones to help NN find board edges */
    planes.push(Bitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    return planes;
}
