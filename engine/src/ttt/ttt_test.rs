#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};
    use std::cmp::Ordering;
    use std::collections::HashSet;

    use crate::game::common::{
        GameBitboard, GameColor, GameMove, GamePlayer, GamePosition, IGame, PlayerRand,
    };
    use crate::ttt::ttt_game::{TttGame, TttMove, TttPosition};

    #[test]
    fn simple_game_and_mate() {
        let to_pos = |s: &str| ttt_position_from_str(s);
        assert!(to_pos("xxxoo____o").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oo_xxx___o").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oo____xxxo").get_winner() == Some(GameColor::Player1));
        assert!(to_pos("oxxo__ox_x").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xox_o_xo_x").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xxo__o_xox").get_winner() == Some(GameColor::Player2));
        assert!(to_pos("xxoooxxxoo").get_winner().is_none());
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
        .map(ttt_position_from_str)
        {
            assert!(pos.get_turn().opposite() == pos.get_flip().get_turn());
            assert!(pos.get_flip().get_flip() == pos);
        }
    }

    #[test]
    fn flip_rand() {
        let seed: u64 = rand::rng().random();
        println!("[{}] Using seed {}", stringify!(flip_rand), seed);
        let mut rand = StdRng::seed_from_u64(seed);

        let games_num = 100;
        for _ in 0..games_num {
            let mut player = PlayerRand::from_seed(rand.next_u64() ^ 0xe4655449311aee87);
            let mut game = TttGame::new();

            while !game.is_over() {
                let pos = *game.get_position();
                let pos_t = pos.get_flip();

                /* Assert flip of flip is original */
                assert!(pos == pos_t.get_flip());

                /* Assert flip of moves of flip are original moves */
                let moves: HashSet<TttMove> = HashSet::from_iter(pos.get_legal_moves());
                let moves_tt: HashSet<TttMove> =
                    HashSet::from_iter(pos_t.get_legal_moves().into_iter().map(|m| m.get_flip()));
                assert!(moves == moves_tt);

                /* Assert game result is the same */
                assert!(pos.is_over() == pos_t.is_over());
                if pos.is_over() {
                    assert!(pos.get_winner() == pos_t.get_winner().map(|w| w.opposite()));
                }

                let next_move =
                    <_ as GamePlayer<TttGame>>::next_move(&mut player, game.get_position())
                        .unwrap();
                game.play_single_turn(next_move);
            }
        }
    }

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
}
