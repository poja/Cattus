#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};
    use std::collections::HashSet;

    use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
    use crate::game::common::GameMove;
    use crate::game::common::GamePlayer;
    use crate::game::common::IGame;
    use crate::game::common::{GameColor, GamePosition, PlayerRand};

    #[test]
    fn simple_game_and_mate() {
        let mut pos = ChessPosition::new();

        let moves = vec![
            "e4", "e5", "d4", "exd4", "Qxd4", "Nc6", "Qa4", "a6", "Bg5", "h6", "Bc4", "Rb8", "Qb3",
            "Ra8", "Bxf7",
        ];
        for move_str in moves {
            assert!(!pos.is_over());
            let m = ChessMove::from_san(&pos, move_str).unwrap();
            pos = pos.get_moved_position(m);
        }
        assert!(pos.is_over());
        assert!(pos.get_winner().unwrap().eq(&GameColor::Player1));
    }

    #[test]
    fn fifty_rule_count() {
        let mut pos = ChessPosition::new();

        #[rustfmt::skip]
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
            let m = ChessMove::from_san(&pos, move_str).unwrap();
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
        .map(ChessPosition::from_str)
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
            let mut game = ChessGame::new();

            while !game.is_over() {
                let pos = *game.get_position();
                let pos_t = pos.get_flip();

                /* Assert flip of flip is original */
                assert!(pos == pos_t.get_flip());

                /* Assert flip of moves of flip are original moves */
                let moves: HashSet<ChessMove> = HashSet::from_iter(pos.get_legal_moves());
                let moves_tt: HashSet<ChessMove> =
                    HashSet::from_iter(pos_t.get_legal_moves().into_iter().map(|m| m.get_flip()));
                assert!(moves == moves_tt);

                /* Assert game result is the same */
                assert!(pos.is_over() == pos_t.is_over());
                if pos.is_over() {
                    assert!(pos.get_winner() == pos_t.get_winner().map(|w| w.opposite()));
                }

                let next_move = <_ as GamePlayer<ChessGame>>::next_move(&mut player, &pos).unwrap();
                game.play_single_turn(next_move);
            }
        }
    }
}
