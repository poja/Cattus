#[cfg(test)]
mod tests {
    use crate::chess::chess_game::ChessPosition;
    use crate::chess::net::net_trivial::TrivialNet;
    use crate::game::mcts::ValueFunction;

    #[test]
    fn basic_evaluate() {
        let net = TrivialNet {};

        let pos1 =
            ChessPosition::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let pos2 =
            ChessPosition::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
        let pos3 =
            ChessPosition::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
        let pos4 =
            ChessPosition::from_fen("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

        assert!(net.evaluate(&pos1).1 == -net.evaluate(&pos3).1);
        assert!(net.evaluate(&pos2).1 == -net.evaluate(&pos4).1);
        assert!(net.evaluate(&pos1).1 > net.evaluate(&pos2).1);
        assert!(-net.evaluate(&pos3).1 > -net.evaluate(&pos4).1);
    }
}
