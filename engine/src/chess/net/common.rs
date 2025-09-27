use crate::chess::chess_game::{ChessBitboard, ChessPosition};
use crate::game::common::GameBitboard;

pub const PLANES_NUM: usize = 18;

pub fn position_to_planes(pos: &ChessPosition) -> Vec<ChessBitboard> {
    let mut planes = [ChessBitboard::from_raw(0); PLANES_NUM];
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
    planes[0] = ChessBitboard::from(pawns & white);
    planes[1] = ChessBitboard::from(knights & white);
    planes[2] = ChessBitboard::from(bishops & white);
    planes[3] = ChessBitboard::from(rooks & white);
    planes[4] = ChessBitboard::from(queens & white);
    planes[5] = ChessBitboard::from(kings & white);
    planes[6] = ChessBitboard::from(pawns & black);
    planes[7] = ChessBitboard::from(knights & black);
    planes[8] = ChessBitboard::from(bishops & black);
    planes[9] = ChessBitboard::from(rooks & black);
    planes[10] = ChessBitboard::from(queens & black);
    planes[11] = ChessBitboard::from(kings & black);

    /* 4 planes of castling rights */
    let white_cr = b.castle_rights(chess::Color::White);
    let black_cr = b.castle_rights(chess::Color::Black);
    planes[12] = ChessBitboard::new_with_all(white_cr.has_kingside());
    planes[13] = ChessBitboard::new_with_all(white_cr.has_queenside());
    planes[14] = ChessBitboard::new_with_all(black_cr.has_kingside());
    planes[15] = ChessBitboard::new_with_all(black_cr.has_queenside());

    /* A plane of en passant */
    planes[16] = ChessBitboard::from_raw(b.en_passant().map_or(0, |s| 1 << s.to_index()));

    /* A plane with all ones to help NN find board edges */
    planes[17] = ChessBitboard::new_with_all(true);

    planes.to_vec()
}
