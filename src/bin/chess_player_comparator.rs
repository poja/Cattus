use clap::Parser;
use rl::chess::chess_game::ChessGame;
use rl::chess::net::two_headed_net::TwoHeadedNet;
use rl::game::mcts::{MCTSPlayer, ValueFunction};
use rl::game::players_compare::{PlayerBuilder, PlayerComparator};

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    model1_path: String,
    #[clap(long)]
    model2_path: String,
    #[clap(long, default_value = "10")]
    games_num: u32,
    #[clap(long)]
    result_file: String,
    #[clap(long, default_value = "100")]
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
    #[clap(long, default_value = "1")]
    threads: u32,
}

struct Builder {
    model_path: String,
    sim_count: u32,
    explore_factor: f32,
}

impl Builder {
    fn new(model_path: String, sim_count: u32, explore_factor: f32) -> Self {
        Self {
            model_path: model_path,
            sim_count: sim_count,
            explore_factor: explore_factor,
        }
    }
}

impl PlayerBuilder<ChessGame> for Builder {
    fn new_player(&self) -> MCTSPlayer<ChessGame> {
        let value_func: Box<dyn ValueFunction<ChessGame>> =
            Box::new(TwoHeadedNet::new(&self.model_path));
        return MCTSPlayer::new_custom(self.sim_count, self.explore_factor, value_func);
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let player1_builder = Box::new(Builder::new(
        args.model1_path,
        args.sim_count,
        args.explore_factor,
    ));
    let player2_builder = Box::new(Builder::new(
        args.model2_path,
        args.sim_count,
        args.explore_factor,
    ));

    let comparator = PlayerComparator::new(player1_builder, player2_builder, args.threads);
    return comparator.compare_players(args.games_num, &args.result_file);
}
