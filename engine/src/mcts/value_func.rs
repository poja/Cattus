pub trait ValueFunction<Game: crate::game::Game>: Sync + Send {
    /// Evaluate a position
    ///
    /// position - The position to evaluate
    ///
    /// Returns a tuple of a scalar value score of the position and per-move scores/probabilities.
    /// The scalar is the current position value in range [-1,1]. 1 if player1 is winning and -1 if
    /// player2 is winning The per-move probabilities should have a sum of 1, greater value is a
    /// better move
    fn evaluate(&self, position: &Game::Position) -> (Vec<(Game::Move, f32)>, f32);
}

// pub struct ValueFunctionRand {
//     rand: StdRng,
// }
// impl<Game: IGame> ValueFunction<Game> for ValueFunctionRand {
//     fn evaluate(&self, position: &Game::Position) -> (Vec<(Game::Move, f32)>, f32) {
//         let winner = if position.is_over() {
//             position.status()
//         } else {
//             // Play randomly and return the simulation game result
//             let mut player1 = PlayerRand::new();
//             let mut player2 = PlayerRand::new();
//             let mut game = Game::from_position(*position);

//             let (_final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
//             winner
//         };
//         let val = match winner {
//             Some(color) => {
//                 if color == position.turn() {
//                     1.0
//                 } else {
//                     -1.0
//                 }
//             }
//             None => 0.0,
//         };

//         /* We don't have anything smart to say per move */
//         /* Assign uniform probabilities to all legal moves */
//         let moves = position.legal_moves();
//         let move_prob = 1.0 / moves.len() as f32;
//         let moves_probs = moves.iter().map(|m| (*m, move_prob)).collect_vec();

//         (moves_probs, val)
//     }
// }
