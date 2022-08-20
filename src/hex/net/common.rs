use crate::game::common::{GameColor, GamePosition, IGame};
use crate::hex::hex_game::HexGame;
use crate::hex::hex_game::HexPosition;
use itertools::Itertools;

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
            .map(|((r, c), p)| ((*c, *r), *p))
            .collect_vec();

        return (val, moves_probs);
    }
}
