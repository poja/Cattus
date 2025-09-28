use cattus_self_play::hex_self_player;

fn main() -> std::io::Result<()> {
    hex_self_player::run_main::<9>()
}
