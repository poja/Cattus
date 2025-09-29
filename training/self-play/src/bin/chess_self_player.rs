use cattus::chess::ChessGame;
use cattus_self_play::self_play_cmd::run_main;
use cattus_self_play::serialize::chess::ChessSerializer;

fn main() -> std::io::Result<()> {
    run_main::<ChessGame>(Box::new(ChessSerializer))
}
