use rl::{hex::uxi, game::common::PlayerRand};

fn main() {
    let mut player = PlayerRand::new();
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
