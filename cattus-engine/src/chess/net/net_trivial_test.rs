#[cfg(test)]
mod tests {
    use crate::chess::chess_game::ChessPosition;
    use crate::chess::net::net_trivial::TrivialNet;
    use crate::game::mcts::ValueFunction;

    #[test]
    fn basic_evaluate() {
        let net = TrivialNet {};

        let pos1 = ChessPosition::from_str(
            &"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
        );
        let pos2 = ChessPosition::from_str(
            &"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1".to_string(),
        );
        let pos3 = ChessPosition::from_str(
            &"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1".to_string(),
        );
        let pos4 = ChessPosition::from_str(
            &"r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1".to_string(),
        );

        assert!(net.evaluate(&pos1).0 == -net.evaluate(&pos3).0);
        assert!(net.evaluate(&pos2).0 == -net.evaluate(&pos4).0);
        assert!(net.evaluate(&pos1).0 > net.evaluate(&pos2).0);
        assert!(-net.evaluate(&pos3).0 > -net.evaluate(&pos4).0);
    }
}
