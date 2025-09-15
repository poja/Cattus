use cattus::game::common::PlayerRand;
use cattus::hex::uxi;

fn main() {
    cattus::util::init_globals();
    let player = Box::new(PlayerRand::new());
    let mut engine = uxi::UXIEngine::new(player);
    engine.run();
}
