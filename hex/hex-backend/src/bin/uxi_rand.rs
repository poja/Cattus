use hex_backend::simple_players;
use hex_backend::uxi;

fn main() {
    let mut player = simple_players::HexPlayerRand::new();
    uxi::UXI::run_engine(&mut player);
}
