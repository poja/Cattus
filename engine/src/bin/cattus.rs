use cattus::chess::net::stockfish::StockfishNet;
use cattus::chess::uci::UCI;
use cattus::game::mcts::MctsParams;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    sim_num: u32,
}

fn main() {
    cattus::util::init_globals(None);

    let args = Args::parse();

    let mut uci = UCI::new(MctsParams::new(args.sim_num, Arc::new(StockfishNet {})));
    uci.run();
}
