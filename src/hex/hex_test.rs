#[cfg(test)]
mod tests {
    use crate::game::common::Bitboard;
    use crate::game::common::{GameColor, GamePosition};
    use crate::hex::hex_game::{HexBitboard, HexPosition, BOARD_SIZE};

    fn string_to_position(s: &str, starting_player: GameColor) -> HexPosition {
        let mut board_red = HexBitboard::new();
        let mut board_blue = HexBitboard::new();
        let mut idx = 0;
        for tile in s.chars() {
            if idx >= BOARD_SIZE * BOARD_SIZE {
                panic!("Too many chars in position string");
            }
            match tile {
                'e' => {}
                'r' => board_red.set(idx, true),
                'b' => board_blue.set(idx, true),
                unknown_tile => {
                    panic!("Unknown tile: '{}'", unknown_tile);
                }
            };
            idx += 1;
        }
        if idx != BOARD_SIZE * BOARD_SIZE {
            panic!("Too few chars in position string");
        }

        return HexPosition::new_from_board(board_red, board_blue, starting_player);
    }

    #[test]
    fn short_diagonal_wins() {
        let pos = string_to_position(
            "reeeeeeeeee\
				ereeeeeeeee\
				eereeeeeeee\
				eeereeeeeee\
				eeeereeeeee\
				eeeeereeeee\
				eeeeeereeee\
				eeeeeeereee\
				eeeeeeeeree\
				eeeeeeeeere\
				eeeeeeeeeer",
            GameColor::Player2,
        );
        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player1));

        let pos = string_to_position(
            "beeeeeeeeee\
				ebeeeeeeeee\
				eebeeeeeeee\
				eeebeeeeeee\
				eeeebeeeeee\
				eeeeebeeeee\
				eeeeeebeeee\
				eeeeeeebeee\
				eeeeeeeebee\
				eeeeeeeeebe\
				eeeeeeeeeeb",
            GameColor::Player1,
        );

        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player2));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let pos = string_to_position(
            "eeeeeeeeeee\
				ereeeeeeeee\
				eereeeeeeee\
				eeereeeeeee\
				eeeereeeeee\
				eeeeereeeee\
				eeeeeereeee\
				eeeeeeereee\
				eeeeeeeeree\
				eeeeeeeeere\
				eeeeeeeeeer",
            GameColor::Player2,
        );
        assert!(!pos.is_over());

        let pos = string_to_position(
            "beeeeeeeeee\
				ebeeeeeeeee\
				eebeeeeeeee\
				eeebeeeeeee\
				eeeebeeeeee\
				eeeeebeeeee\
				eeeeeebeeee\
				eeeeeeebeee\
				eeeeeeeebee\
				eeeeeeeeebe\
				eeeeeeeeeee",
            GameColor::Player1,
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let pos = string_to_position(
            "eeeeeeeeeer\
				eeeeeeeeere\
				eeeeeeeeree\
				eeeeeeereee\
				eeeeeereeee\
				eeeeereeeee\
				eeeereeeeee\
				eeereeeeeee\
				eereeeeeeee\
				ereeeeeeeee\
				reeeeeeeeee",
            GameColor::Player2,
        );
        assert!(!pos.is_over());

        let pos = string_to_position(
            "eeeeeeeeeeb\
				eeeeeeeeebe\
				eeeeeeeebee\
				eeeeeeebeee\
				eeeeeebeeee\
				eeeeebeeeee\
				eeeebeeeeee\
				eeebeeeeeee\
				eebeeeeeeee\
				ebeeeeeeeee\
				beeeeeeeeee",
            GameColor::Player1,
        );
        assert!(!pos.is_over());
    }
}
