use crate::chess::chess_game::{ChessBitboard, ChessGame, ChessMove, ChessPosition};
use crate::game::common::{Bitboard, GameColor, GamePosition, IGame};
use itertools::Itertools;

pub const PLANES_NUM: usize = 18;
pub const MOVES_NUM: usize = 1880;

pub fn flip_pos_if_needed(pos: ChessPosition) -> (ChessPosition, bool) {
    if pos.get_turn() == GameColor::Player1 {
        return (pos, false);
    } else {
        return (ChessPosition::flip_of(&pos), true);
    }
}

pub fn flip_score_if_needed(
    net_res: (f32, Vec<(<ChessGame as IGame>::Move, f32)>),
    pos_flipped: bool,
) -> (f32, Vec<(<ChessGame as IGame>::Move, f32)>) {
    if !pos_flipped {
        return net_res;
    } else {
        let (val, moves_probs) = net_res;

        /* Flip scalar value */
        let val = -val;

        /* Flip moves */
        let flip_rank = |r: chess::Rank| chess::Rank::from_index(7 - r.to_index());
        let flip_file = |f: chess::File| f;
        let flip_square = |s: chess::Square| {
            chess::Square::make_square(flip_rank(s.get_rank()), flip_file(s.get_file()))
        };
        let moves_probs = moves_probs
            .iter()
            .map(|(m, p)| {
                let s = flip_square(m.get_raw().get_source());
                let d = flip_square(m.get_raw().get_dest());
                let promotion = m.get_raw().get_promotion();
                (ChessMove::new(chess::ChessMove::new(s, d, promotion)), *p)
            })
            .collect_vec();

        return (val, moves_probs);
    }
}

pub fn position_to_planes(pos: &ChessPosition) -> Vec<ChessBitboard> {
    let mut planes = Vec::new();
    let b = pos.get_raw_board();

    /* 12 planes of pieces */
    let pawns = b.pieces(chess::Piece::Pawn);
    let knights = b.pieces(chess::Piece::Knight);
    let bishops = b.pieces(chess::Piece::Bishop);
    let rooks = b.pieces(chess::Piece::Rook);
    let queens = b.pieces(chess::Piece::Queen);
    let kings = b.pieces(chess::Piece::King);
    let white = b.color_combined(chess::Color::White);
    let black = b.color_combined(chess::Color::Black);
    let mut planes_raw = Vec::new();
    planes_raw.push(pawns & white);
    planes_raw.push(knights & white);
    planes_raw.push(bishops & white);
    planes_raw.push(rooks & white);
    planes_raw.push(queens & white);
    planes_raw.push(kings & white);
    planes_raw.push(pawns & black);
    planes_raw.push(knights & black);
    planes_raw.push(bishops & black);
    planes_raw.push(rooks & black);
    planes_raw.push(queens & black);
    planes_raw.push(kings & black);
    for p in planes_raw {
        planes.push(ChessBitboard::from_raw(p.0));
    }

    /* 4 planes of castling rights */
    let white_cr = b.castle_rights(chess::Color::White);
    let black_cr = b.castle_rights(chess::Color::Black);
    planes.push(ChessBitboard::new_with_all(white_cr.has_kingside()));
    planes.push(ChessBitboard::new_with_all(white_cr.has_queenside()));
    planes.push(ChessBitboard::new_with_all(black_cr.has_kingside()));
    planes.push(ChessBitboard::new_with_all(black_cr.has_queenside()));

    /* A plane of en passant */
    planes.push(ChessBitboard::from_raw(
        b.en_passant().map_or(0, |s| 1 << s.to_index()),
    ));

    /* A plane with all ones to help NN find board edges */
    planes.push(ChessBitboard::new_with_all(true));

    assert!(planes.len() == PLANES_NUM);
    return planes;
}
