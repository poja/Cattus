use clap::Parser;
use rl::game::mcts::{MCTSPlayer, ValueFunction};
use rl::game::self_play::{PlayerBuilder, SelfPlayRunner};
use rl::tictactoe::net::encoder::SimpleEncoder;
use rl::tictactoe::net::two_headed_net::TwoHeadedNet;
use rl::tictactoe::tictactoe_game::TicTacToeGame;

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

struct Builder {
    net_type: String,
    model_path: String,
    sim_count: u32,
    explore_factor: f32,
}

impl Builder {
    fn new(net_type: String, model_path: String, sim_count: u32, explore_factor: f32) -> Self {
        Self {
            net_type: net_type,
            model_path: model_path,
            sim_count: sim_count,
            explore_factor: explore_factor,
        }
    }
}

impl PlayerBuilder<TicTacToeGame> for Builder {
    fn new_player(&self) -> MCTSPlayer<TicTacToeGame> {
        if self.net_type != "two_headed_net" {
            panic!("unsupported net type: {}", self.net_type);
        }
        let value_func: Box<dyn ValueFunction<TicTacToeGame>> =
            Box::new(TwoHeadedNet::new(&self.model_path));
        return MCTSPlayer::new_custom(self.sim_count, self.explore_factor, value_func);
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.net_type != "two_headed_net" {
        panic!("unsupported net type: {}", args.net_type);
    }

    let player_builder = Box::new(Builder::new(
        args.net_type,
        args.model_path,
        args.sim_count,
        args.explore_factor,
    ));

    let encoder = Box::new(SimpleEncoder::new());
    let trainer = SelfPlayRunner::new(encoder);
    return trainer.generate_data(player_builder, args.games_num, &args.out_dir);
}
