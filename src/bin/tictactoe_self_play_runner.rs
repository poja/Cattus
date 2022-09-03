use clap::Parser;
use rl::game::mcts::MCTSPlayer;
use rl::game::self_play::SelfPlayRunner;
use rl::tictactoe::net::encoder::SimpleEncoder;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;


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
    if args.net_type != "two_headed_net" {
        panic!("unsupported net type: {}", args.net_type);
    }

    let mut value_func_net_two_headed = TwoHeadedNet::new(args.model_path);
    let mut player = MCTSPlayer::new_custom(args.sim_count, args.explore_factor, &mut 
        value_func_net_two_headed);

    let mut encoder = SimpleEncoder::new();
    let trainer = SelfPlayRunner::new(&mut encoder);

    return trainer.generate_data(&mut player, args.games_num, &args.out_dir);
}
