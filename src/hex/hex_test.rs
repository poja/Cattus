#[cfg(test)]
mod tests {
    use crate::game::common::{GameColor, GamePosition};
    use crate::hex::hex_game::HexPosition;
    use crate::hex::hex_game::Hexagon;

    #[test]
    fn short_diagonal_wins() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(GameColor::Player1);
        let b = Hexagon::Full(GameColor::Player2);

        let pos = HexPosition::new_from_board(
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
            GameColor::Player2,
        );
        assert!(pos.get_winner() == Some(GameColor::Player1));

        let pos = HexPosition::new_from_board(
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
            GameColor::Player1,
        );
        assert!(pos.get_winner() == Some(GameColor::Player2));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(GameColor::Player1);
        let b = Hexagon::Full(GameColor::Player2);

        let pos = HexPosition::new_from_board(
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
            GameColor::Player2,
        );
        assert!(!pos.is_over());

        let pos = HexPosition::new_from_board(
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
            GameColor::Player1,
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let e = Hexagon::Empty;
        let r = Hexagon::Full(GameColor::Player1);
        let b = Hexagon::Full(GameColor::Player2);

        let pos = HexPosition::new_from_board(
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
            GameColor::Player2,
        );
        assert!(!pos.is_over());

        let pos = HexPosition::new_from_board(
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
            GameColor::Player1,
        );
        assert!(!pos.is_over());
    }
}
