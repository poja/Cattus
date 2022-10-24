use clap::Parser;
use cattus::chess::chess_game::ChessGame;
use cattus::chess::net::net_stockfish::StockfishNet;
use cattus::chess::uci::UCI;
use cattus::game::mcts::MCTSPlayer;
use cattus::utils::Builder;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    sim_num: u32,
}

struct PlayerBuilder {
    sim_num: u32,
}
impl Builder<MCTSPlayer<ChessGame>> for PlayerBuilder {
    fn build(&self) -> MCTSPlayer<ChessGame> {
        let value_func = Box::new(StockfishNet {});
        MCTSPlayer::new(self.sim_num, value_func)
    }
}

fn main() {
    let args = Args::parse();

    let builder = Box::new(PlayerBuilder {
        sim_num: args.sim_num,
    });

    let mut uci = UCI::new(builder);
    uci.run();
}
