use clap::Parser;
use rl::chess::chess_game::ChessGame;
use rl::chess::net::net_stockfish::StockfishNet;
use rl::chess::uci::{Builder, UCI};
use rl::game::mcts::MCTSPlayer;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    sim_count: u32,
}

struct PlayerBuilder {
    sim_count: u32,
}
impl Builder<MCTSPlayer<ChessGame>> for PlayerBuilder {
    fn build(&self) -> MCTSPlayer<ChessGame> {
        let value_func = Box::new(StockfishNet::new());
        return MCTSPlayer::new_custom(self.sim_count, 1.41421, value_func);
    }
}

fn main() {
    let args = Args::parse();

    let builder = Box::new(PlayerBuilder {
        sim_count: args.sim_count,
    });

    let mut uci = UCI::new(builder);
    uci.run();
}
