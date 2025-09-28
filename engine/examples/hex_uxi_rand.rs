use cattus::game::player::PlayerRand;
use cattus::hex::uxi;

fn main() {
    cattus::util::init_globals();
    let player = Box::new(PlayerRand::new());
    let mut engine = uxi::UxiEngine::new(player);
    engine.run();
}
