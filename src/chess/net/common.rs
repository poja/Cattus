use crate::chess::chess_game::{ChessBitboard, ChessPosition};
use crate::game::common::Bitboard;

pub const PLANES_NUM: usize = 18;
pub const MOVES_NUM: usize = 1880;

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
