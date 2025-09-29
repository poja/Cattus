use cattus_self_play::self_play_cmd::run_main;
use cattus_self_play::serialize::ttt::TttSerializer;

fn main() -> std::io::Result<()> {
    run_main(Box::new(TttSerializer))
}
