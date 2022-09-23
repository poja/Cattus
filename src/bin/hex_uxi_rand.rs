use rl::{game::common::PlayerRand, hex::uxi};

fn main() {
    let player = Box::new(PlayerRand::new());
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
