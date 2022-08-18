use clap::Parser;
use rl::game::mcts::MCTSPlayer;
use rl::game::self_play::SelfPlayRunner;
use rl::hex::net::encoder::SimpleEncoder;
use rl::hex::net::scalar_value_net::ScalarValNet;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model: String,
    #[clap(long, default_value = "10")]
    games_num: u32,
    #[clap(long)]
    out_dir: String,
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    // TODO add arg for network type
    let mut encoder = SimpleEncoder::new();
    let trainer = SelfPlayRunner::new(&mut encoder);
    let mut value_func = ScalarValNet::new(args.model);
    let mut player = MCTSPlayer::new_custom(args.sim_count, args.explore_factor, &mut value_func);
    return trainer.generate_data(&mut player, args.games_num, &args.out_dir);
}
