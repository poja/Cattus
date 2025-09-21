use cattus::game::common::PlayerRand;
use cattus::hex::uxi;

fn main() {
    cattus::util::init_globals(None);
    let player = Box::new(PlayerRand::new());
    let mut engine = uxi::UxiEngine::new(player);
    engine.run();
}
