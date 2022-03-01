use hex_backend::mcts;
use hex_backend::uxi;

fn main() {
    let mut player = mcts::MCTSPlayer::new();
    uxi::UXI::run_engine(&mut player);
}
