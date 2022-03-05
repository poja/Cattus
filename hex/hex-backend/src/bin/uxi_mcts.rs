use hex_backend::mcts;
use hex_backend::uxi;

fn main() {
    let mut player = mcts::MCTSPlayer::new();
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
