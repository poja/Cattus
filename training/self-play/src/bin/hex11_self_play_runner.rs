use cattus_self_play::hex_self_play_cmd;

fn main() -> std::io::Result<()> {
    hex_self_play_cmd::run_main::<11>()
}
