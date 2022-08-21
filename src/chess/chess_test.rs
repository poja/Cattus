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
    fn fifth_rule_count() {
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
            println!("{:?}", pos.fifth_rule_count);
        }
        assert!(pos.is_over());
        assert!(pos.get_winner().is_none());
    }
}
