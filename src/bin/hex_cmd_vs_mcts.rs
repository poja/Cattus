use clap::Parser;
use std::sync::Arc;

use cattus::game::cache::ValueFuncCache;
use cattus::game::common::{GameColor, GamePosition, IGame};
use cattus::game::mcts::MCTSPlayer;
use cattus::hex::cmd_player::HexPlayerCmd;
use cattus::hex::hex_game::HexGame;
use cattus::hex::net::two_headed_net::TwoHeadedNet;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long, default_value = "11")]
    board_size: u32,
    #[clap(long)]
    model_path: String,
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
    #[clap(long, default_value = "1.0")]
    temperature_policy: String,
    #[clap(long, default_value = "0.0")]
    prior_noise_alpha: f32,
    #[clap(long, default_value = "0.0")]
    prior_noise_epsilon: f32,
    #[clap(long, default_value = "100000")]
    cache_size: usize,
}

fn run_main<const BOARD_SIZE: usize>(args: Args) {
    let mut player1 = HexPlayerCmd {};

    let cache = Arc::new(ValueFuncCache::new(args.cache_size));
    let value_func = Box::new(TwoHeadedNet::<BOARD_SIZE, true>::with_cache(
        &args.model_path,
        cache,
    ));
    let mut player2 = MCTSPlayer::new_custom(
        args.sim_num,
        args.explore_factor,
        args.prior_noise_alpha,
        args.prior_noise_epsilon,
        value_func,
    );

    let mut game = HexGame::<BOARD_SIZE>::new();

    let (final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
    println!(
        "The winner is: {}, details below:",
        match winner {
            None => String::from("Tie"),
            Some(GameColor::Player1) => String::from("CMD player"),
            Some(GameColor::Player2) => String::from("MCTS player"),
        }
    );
    final_pos.print();
}

fn main() {
    let args = Args::parse();
    match args.board_size {
        5 => run_main::<5>(args),
        7 => run_main::<7>(args),
        9 => run_main::<9>(args),
        11 => run_main::<11>(args),
        other => panic!("unsupported hex size: {other}"),
    };
}
