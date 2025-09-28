use std::cmp::Ordering;

use cattus::game::{Bitboard, Game as _, GameColor, Position};
use cattus::hex::{HexBitboard, HexPosition};
use cattus::ttt::{TttGame, TttPosition};

pub fn ttt_position_from_str(s: &str) -> TttPosition {
    assert_eq!(
        s.chars().count(),
        TttGame::BOARD_SIZE * TttGame::BOARD_SIZE + 1,
        "unexpected string length"
    );
    let mut pos = TttPosition::new();
    for (idx, c) in s.chars().enumerate() {
        match idx.cmp(&(TttGame::BOARD_SIZE * TttGame::BOARD_SIZE)) {
            Ordering::Less => match c {
                'x' => pos.board_x.set(idx, true),
                'o' => pos.board_o.set(idx, true),
                '_' => {}
                _ => panic!("unknown board char: {:?}", c),
            },
            Ordering::Equal => {
                pos.turn = match c {
                    'x' => GameColor::Player1,
                    'o' => GameColor::Player2,
                    _ => panic!("unknown turn char: {:?}", c),
                }
            }
            Ordering::Greater => panic!("too many turn chars: {:?}", c),
        }
    }
    pos.check_winner();
    pos
}

pub fn hex_position_from_str<const BOARD_SIZE: usize>(s: &str) -> HexPosition<BOARD_SIZE> {
    assert_eq!(
        s.chars().count(),
        BOARD_SIZE * BOARD_SIZE + 1,
        "unexpected string length"
    );

    let mut board_red = HexBitboard::new();
    let mut board_blue = HexBitboard::new();
    let mut turn = None;
    for (idx, c) in s.chars().enumerate() {
        match idx.cmp(&(BOARD_SIZE * BOARD_SIZE)) {
            Ordering::Less => match c {
                'e' => {}
                'r' => board_red.set(idx, true),
                'b' => board_blue.set(idx, true),
                _ => panic!("unknown board char: {:?}", c),
            },
            Ordering::Equal => {
                turn = Some(match c {
                    'r' => GameColor::Player1,
                    'b' => GameColor::Player2,
                    _ => panic!("unknown turn char: {:?}", c),
                })
            }
            Ordering::Greater => panic!("Too many chars in position string"),
        }
    }

    HexPosition::new_from_board(board_red, board_blue, turn.unwrap())
}
