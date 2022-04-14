use rl::hex::simple_players;
use rl::hex::uxi;

fn main() {
    let mut player = simple_players::PlayerRand::new();
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
