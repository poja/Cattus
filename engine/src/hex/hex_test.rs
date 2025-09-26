#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};
    use std::cmp::Ordering;
    use std::collections::HashSet;

    use crate::game::common::{GameBitboard, GameColor, GameMove, GamePlayer, GamePosition, IGame, PlayerRand};
    use crate::hex::hex_game::{HexBitboard, HexGameStandard, HexPosition};

    type HexStandardPosition = <HexGameStandard as IGame>::Position;

    #[test]
    fn short_diagonal_wins() {
        let pos: HexStandardPosition = hex_position_from_str(
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
            eeeeeeeeeer\
            b",
        );
        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player1));

        let pos: HexStandardPosition = hex_position_from_str(
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
            eeeeeeeeeeb\
            r",
        );

        assert!(pos.is_over());
        assert!(pos.get_winner() == Some(GameColor::Player2));
    }

    #[test]
    fn almost_short_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
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
            eeeeeeeeeer\
            b",
        );
        assert!(!pos.is_over());

        let pos: HexStandardPosition = hex_position_from_str(
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
            eeeeeeeeeee\
            r",
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn long_diagonal_doesnt_win() {
        let pos: HexStandardPosition = hex_position_from_str(
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
            reeeeeeeeee\
            b",
        );
        assert!(!pos.is_over());

        let pos: HexStandardPosition = hex_position_from_str(
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
            beeeeeeeeee\
            r",
        );
        assert!(!pos.is_over());
    }

    #[test]
    fn flip() {
        let pos: HexStandardPosition = hex_position_from_str(
            "eebeeeeeeer\
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
            b",
        );
        assert!(pos.get_turn() == GameColor::Player2);
        assert!(pos.get_flip().get_turn() == GameColor::Player1);
        assert!(pos.get_flip().get_flip() == pos);
    }

    #[test]
    fn flip_rand() {
        let seed: u64 = rand::rng().random();
        println!("[{}] Using seed {}", stringify!(flip_rand), seed);
        let mut rand = StdRng::seed_from_u64(seed);

        let games_num = 100;
        for _ in 0..games_num {
            let mut player = PlayerRand::from_seed(rand.next_u64() ^ 0xe4655449311aee87);
            let mut game = HexGameStandard::new();

            while !game.is_over() {
                let pos = *game.position();
                let pos_t = pos.get_flip();

                /* Assert flip of flip is original */
                assert!(pos == pos_t.get_flip());

                /* Assert flip of moves of flip are original moves */
                type Move = <HexGameStandard as IGame>::Move;
                let moves: HashSet<Move> = HashSet::from_iter(pos.get_legal_moves());
                let moves_tt: HashSet<Move> =
                    HashSet::from_iter(pos_t.get_legal_moves().into_iter().map(|m| m.get_flip()));
                assert!(moves == moves_tt);

                /* Assert game result is the same */
                assert!(pos.is_over() == pos_t.is_over());
                if pos.is_over() {
                    assert!(pos.get_winner() == pos_t.get_winner().map(|w| w.opposite()));
                }

                let next_move = <_ as GamePlayer<HexGameStandard>>::next_move(&mut player, game.pos_history()).unwrap();
                game.play_single_turn(next_move);
            }
        }
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
}
