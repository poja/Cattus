use clap::Parser;
use rl::game::mcts::{MCTSPlayer, ValueFunction};
use rl::game::self_play::SelfPlayRunner;
use rl::hex::hex_game::HexGame;
use rl::hex::net::encoder::SimpleEncoder;
use rl::hex::net::scalar_value_net::ScalarValNet;
use rl::hex::net::two_headed_net::TwoHeadedNet;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    net_type: String,
    #[clap(long)]
    model_path: String,
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

    let value_func: Box<dyn ValueFunction<HexGame>>;
    if args.net_type == "scalar_net" {
        value_func = Box::new(ScalarValNet::new(&args.model_path));
    } else if args.net_type == "two_headed_net" {
        value_func = Box::new(TwoHeadedNet::new(&args.model_path));
    } else {
        panic!("unsupported net type: {}", args.net_type);
    }
    let mut player = MCTSPlayer::new_custom(args.sim_count, args.explore_factor, value_func);

    let encoder = Box::new(SimpleEncoder::new());
    let trainer = SelfPlayRunner::new(encoder);

    return trainer.generate_data(&mut player, args.games_num, &args.out_dir);
}
