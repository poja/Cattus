use cattus::{game::common::PlayerRand, hex::uxi};

fn main() {
    let player = Box::new(PlayerRand {});
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
