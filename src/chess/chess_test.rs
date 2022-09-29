#[cfg(test)]
mod tests {
    use crate::chess::chess_game::{ChessMove, ChessPosition};
    use crate::game::common::{GameColor, GamePosition};

    #[test]
    fn simple_game_and_mate() {
        let mut pos = ChessPosition::new();

        let moves = vec![
            "e4", "e5", "d4", "exd4", "Qxd4", "Nc6", "Qa4", "a6", "Bg5", "h6", "Bc4", "Rb8", "Qb3",
            "Ra8", "Bxf7",
        ];
        for move_str in moves {
            assert!(!pos.is_over());
            let m = match ChessMove::from_str(&pos, move_str) {
                Ok(m) => m,
                Err(msg) => {
                    println!("failed: {}", msg);
                    assert!(false);
                    return;
                }
            };
            pos = pos.get_moved_position(m);
        }
        assert!(pos.is_over());
        assert!(pos.get_winner().unwrap().eq(&GameColor::Player1));
    }

    #[test]
    fn fifty_rule_count() {
        let mut pos = ChessPosition::new();

        let moves = vec![
            "e4", "e5",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1", "Ke8",
            "Ke2", "Ke7", "Ke1",
        ];
        for move_str in moves {
            assert!(!pos.is_over());
            let m = match ChessMove::from_str(&pos, move_str) {
                Ok(m) => m,
                Err(msg) => {
                    println!("failed: {}", msg);
                    assert!(false);
                    return;
                }
            };
            pos = pos.get_moved_position(m);
        }
        assert!(pos.is_over());
        assert!(pos.get_winner().is_none());
    }

    #[test]
    fn flip() {
        /* random FEN: */
        /* http://bernd.bplaced.net/fengenerator/fengenerator.html */
        for pos in vec![
            "7r/2B3n1/K5R1/3nPP2/P1k2Pp1/4p1p1/2p4P/8 w - - 0 1",
            "8/5pB1/1P4P1/1p3q2/BK1pP2P/pQ1pP3/7k/8 w - - 0 1",
            "2k5/2b1p1P1/1p5P/P1pnp3/QPK1b2p/8/5r2/8 b - - 0 1",
            "2b5/3N4/2B4p/1KP1R2r/n5p1/3N3P/k2B3b/q7 w - - 0 1",
            "8/2P1r3/4k1p1/4n1p1/3Rq3/P3Pp2/P4PP1/5K1N b - - 0 1",
            "1N6/5pP1/P1pK3P/P3p3/2R3p1/k3pQ2/6r1/6N1 b - - 0 1",
            "1B6/1r6/Q1ppKP2/qP1P1N1k/3p4/1pb5/7p/8 w - - 0 1",
            "3r4/1b2p3/7k/1P3R2/K4nrN/1N5P/n1Pp3P/8 b - - 0 1",
        ]
        .into_iter()
        .map(|fen| ChessPosition::from_str(&fen.to_string()))
        {
            assert!(pos.get_turn().opposite() == pos.get_flip().get_turn());
            assert!(pos.get_flip().get_flip() == pos);
        }
    }
}
