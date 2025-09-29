use cattus::hex::HexGame;
use cattus_self_play::self_play_cmd;
use cattus_self_play::serialize::hex::HexSerializer;

fn main() -> std::io::Result<()> {
    self_play_cmd::run_main::<HexGame<4>>(Box::new(HexSerializer))
}
