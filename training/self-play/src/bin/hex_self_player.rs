use cattus::hex::HEX_STANDARD_BOARD_SIZE;
use cattus_self_play::hex_self_player;

fn main() -> std::io::Result<()> {
    hex_self_player::run_main::<HEX_STANDARD_BOARD_SIZE>()
}
