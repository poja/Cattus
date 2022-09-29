#[cfg(test)]
mod tests {
    use crate::game::common::{GameColor, GamePosition};
    use crate::hex::hex_game::HexPosition;

    #[test]
    fn short_diagonal_wins() {
        let pos = HexPosition::from_str(
            &"reeeeeeeeee\
            ereeeeeeeee\
            eereeeeeeee\
            eeereeeeeee\
            eeeereeeeee\
            eeeeereeeee\
            eeeeeereeee\
            eeeeeeereee\
            eeeeeeeeree\
            eeeeeeeeere\
            eeeeeeeeeer\
            b"
            .to_string(),
        );
        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player1));

        let pos = HexPosition::from_str(
            &"beeeeeeeeee\
            ebeeeeeeeee\
            eebeeeeeeee\
            eeebeeeeeee\
            eeeebeeeeee\
            eeeeebeeeee\
            eeeeeebeeee\
            eeeeeeebeee\
            eeeeeeeebee\
            eeeeeeeeebe\
            eeeeeeeeeeb\
            r"
            .to_string(),
        );

        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player2));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let pos = HexPosition::from_str(
            &"eeeeeeeeeee\
            ereeeeeeeee\
            eereeeeeeee\
            eeereeeeeee\
            eeeereeeeee\
            eeeeereeeee\
            eeeeeereeee\
            eeeeeeereee\
            eeeeeeeeree\
            eeeeeeeeere\
            eeeeeeeeeer\
            b"
            .to_string(),
        );
        assert!(!pos.is_over());

        let pos = HexPosition::from_str(
            &"beeeeeeeeee\
            ebeeeeeeeee\
            eebeeeeeeee\
            eeebeeeeeee\
            eeeebeeeeee\
            eeeeebeeeee\
            eeeeeebeeee\
            eeeeeeebeee\
            eeeeeeeebee\
            eeeeeeeeebe\
            eeeeeeeeeee\
            r"
            .to_string(),
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let pos = HexPosition::from_str(
            &"eeeeeeeeeer\
            eeeeeeeeere\
            eeeeeeeeree\
            eeeeeeereee\
            eeeeeereeee\
            eeeeereeeee\
            eeeereeeeee\
            eeereeeeeee\
            eereeeeeeee\
            ereeeeeeeee\
            reeeeeeeeee\
            b"
            .to_string(),
        );
        assert!(!pos.is_over());

        let pos = HexPosition::from_str(
            &"eeeeeeeeeeb\
            eeeeeeeeebe\
            eeeeeeeebee\
            eeeeeeebeee\
            eeeeeebeeee\
            eeeeebeeeee\
            eeeebeeeeee\
            eeebeeeeeee\
            eebeeeeeeee\
            ebeeeeeeeee\
            beeeeeeeeee\
            r"
            .to_string(),
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn flip() {
        let pos = HexPosition::from_str(
            &"eebeeeeeeer\
        eeeeeeeeeee\
        eeeebeeeree\
        eeeeeeereee\
        eeeeeereeee\
        eeeeereeeee\
        eeeerebeeee\
        eeereeeeeee\
        eereeereeee\
        ereeeeeeeee\
        reeeeebeeee\
        b"
            .to_string(),
        );
        assert!(pos.get_turn() == GameColor::Player2);
        assert!(pos.get_flip().get_turn() == GameColor::Player1);
        assert!(pos.get_flip().get_flip() == pos);
    }
}
