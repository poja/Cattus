use crate::game::common::{GameColor, GamePosition, IGame};
use crate::tictactoe::tictactoe_game::TicTacToeGame;
use crate::tictactoe::tictactoe_game::TicTacToePosition;

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
