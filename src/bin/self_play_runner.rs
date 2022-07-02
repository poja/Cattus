use clap::Parser;
use rl::game_utils::{mcts, train};
use rl::network::simple_network;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "10")]
    games_num: u32,
    #[clap(long)]
    out_dir: String,
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_param_c: f32,
}

fn main() {
    let args = Args::parse();
    let mut encoder = simple_network::SimpleEncoder::new();
    let trainer = train::Trainer::new(&mut encoder);
    let mut player = mcts::MCTSPlayer::new_custom(args.sim_count, args.explore_param_c);
    match trainer.generate_data(&mut player, args.games_num, &args.out_dir) {
        Ok(()) => {}
        Err(e) => println!("error parsing header: {e:?}"),
    };
}
