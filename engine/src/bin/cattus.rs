use cattus::chess::chess_game::ChessGame;
use cattus::chess::net::net_stockfish::StockfishNet;
use cattus::chess::uci::UCI;
use cattus::game::mcts::MctsPlayer;
use cattus::util::Builder;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    sim_num: u32,
}

struct PlayerBuilder {
    sim_num: u32,
}
impl Builder<MctsPlayer<ChessGame>> for PlayerBuilder {
    fn build(&self) -> MctsPlayer<ChessGame> {
        let value_func = Arc::new(StockfishNet {});
        MctsPlayer::new(self.sim_num, value_func)
    }
}

fn main() {
    cattus::util::init_globals(None);

    let args = Args::parse();

    let builder = Box::new(PlayerBuilder {
        sim_num: args.sim_num,
    });

    let mut uci = UCI::new(builder);
    uci.run();
}
