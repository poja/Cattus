#[cfg(test)]
mod tests {
    use crate::hex_game::Hexagon;
    use crate::Color::{Blue, Red};
    use crate::hex_game::HexPosition;
    use crate::simple_players::HexPlayerRand;
    use crate::HexGame;

    #[test]
    fn short_diagonal_wins() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();

        let pos = HexPosition {
            board: [
                [r, e, e, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, e, e, r],
            ],
            turn: Blue,
        };
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(s.winner == Some(Red));

        let pos = HexPosition {
            board: [
                [b, e, e, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, e, e, b],
            ],
            turn: Red,
        };
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(s.winner == Some(Blue));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = modname::HexPlayerRand::new();
        let mut player2 = modname::HexPlayerRand::new();

        let pos = HexPosition {
            board: [
                [e, e, e, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, e, e, r],
            ],
            turn: Blue,
        };
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(!s.is_over);

        let pos = HexPosition {
            board: [
                [b, e, e, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Red,
        };
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(!s.is_over);
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = modname::HexPlayerRand::new();
        let mut player2 = modname::HexPlayerRand::new();

        let pos = HexPosition {
            board: [
                [e, e, e, e, e, e, e, e, e, e, r],
                [e, e, e, e, e, e, e, e, e, r, e],
                [e, e, e, e, e, e, e, e, r, e, e],
                [e, e, e, e, e, e, e, r, e, e, e],
                [e, e, e, e, e, e, r, e, e, e, e],
                [e, e, e, e, e, r, e, e, e, e, e],
                [e, e, e, e, r, e, e, e, e, e, e],
                [e, e, e, r, e, e, e, e, e, e, e],
                [e, e, r, e, e, e, e, e, e, e, e],
                [e, r, e, e, e, e, e, e, e, e, e],
                [r, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Blue,
        };
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(!s.is_over);

        let pos = HexPosition {
            board: [
                [e, e, e, e, e, e, e, e, e, e, b],
                [e, e, e, e, e, e, e, e, e, b, e],
                [e, e, e, e, e, e, e, e, b, e, e],
                [e, e, e, e, e, e, e, b, e, e, e],
                [e, e, e, e, e, e, b, e, e, e, e],
                [e, e, e, e, e, b, e, e, e, e, e],
                [e, e, e, e, b, e, e, e, e, e, e],
                [e, e, e, b, e, e, e, e, e, e, e],
                [e, e, b, e, e, e, e, e, e, e, e],
                [e, b, e, e, e, e, e, e, e, e, e],
                [b, e, e, e, e, e, e, e, e, e, e],
            ],
            turn: Red,
        };
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        s.check_if_over();
        assert!(!s.is_over);
    }
}
