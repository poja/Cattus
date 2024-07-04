use cattus::game::common::PlayerRand;
use cattus::hex::uxi;
use cattus::utils;

fn main() {
    utils::init_globals(None);
    let player = Box::new(PlayerRand::new());
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
