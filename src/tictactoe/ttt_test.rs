#[cfg(test)]
mod tests {
    use crate::game::common::{GamePosition, GameColor};
    use crate::tictactoe::tictactoe_game::TicTacToePosition;

    #[test]
    fn simple_game_and_mate() {
        let to_pos = |s: &str| TicTacToePosition::from_str(&s.to_string());
        assert!(to_pos("xxxoo____o").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oo_xxx___o").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oo____xxxo").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oxxo__ox_x").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xox_o_xo_x").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xxo__o_xox").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xxoooxxxoo").get_winner() == None);
    }

    #[test]
    fn flip() {
        for pos in vec![
            "oxx_o_o__o",
            "o_____xx_o",
            "xx_xx_xo_o",
            "ox___x_xox",
            "_x_o__o_xo",
            "ox__o____x",
            "_o__o_oxxo",
            "__xx_x__ox",
        ]
        .into_iter()
        .map(|s| TicTacToePosition::from_str(&s.to_string()))
        {
            assert!(pos.get_turn().opposite() == pos.get_flip().get_turn());
            assert!(pos.get_flip().get_flip() == pos);
        }
    }
}
