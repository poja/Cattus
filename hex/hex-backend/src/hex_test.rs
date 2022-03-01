#[cfg(test)]
mod tests {
    use crate::hex_game::HexPosition;
    use crate::hex_game::Hexagon;
    use crate::simple_players::HexPlayerRand;
    use crate::Color::{Blue, Red};
    use crate::HexGame;

    #[test]
    fn short_diagonal_wins() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();

        let pos = HexPosition::from_board(
            [
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
            Blue,
        );
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(s.position.get_winner() == Some(Red));

        let pos = HexPosition::from_board(
            [
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
            Red,
        );
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(s.position.get_winner() == Some(Blue));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();

        let pos = HexPosition::from_board(
            [
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
            Blue,
        );
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(!s.position.is_over());

        let pos = HexPosition::from_board(
            [
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
            Red,
        );
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(!s.position.is_over());
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(Red);
        let b = Hexagon::Full(Blue);
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();

        let pos = HexPosition::from_board(
            [
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
            Blue,
        );
        let mut s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(!s.position.is_over());

        let pos = HexPosition::from_board(
            [
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
            Red,
        );
        s = HexGame::from_position(&pos, &mut player1, &mut player2);
        assert!(!s.position.is_over());
    }
}
