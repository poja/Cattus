use hex_backend::simple_players;
use hex_backend::uxi;

fn main() {
    let mut player = simple_players::PlayerRand::new();
    let mut engine = uxi::UXIEngine::new(&mut player);
    engine.run();
}
