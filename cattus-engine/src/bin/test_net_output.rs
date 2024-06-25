use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::net::{self, TwoHeadedNetBase};
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::utils;
use cattus::{chess, hex, ttt};
use clap::Parser;
use itertools::Itertools;
use std::fs;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    game: String,
    #[clap(long)]
    position: String,
    #[clap(long)]
    model_path: String,
    #[clap(long)]
    outfile: String,
    #[clap(long, default_value = "1")]
    repeat: u32,
}

fn main() -> std::io::Result<()> {
    utils::init_python();

    let args = Args::parse();

    let outputs = match args.game.as_str() {
        "tictactoe" => run_net_tictactoe(&args),
        "hex4" => run_net_hex::<4>(&args),
        "hex5" => run_net_hex::<5>(&args),
        "hex7" => run_net_hex::<7>(&args),
        "hex9" => run_net_hex::<9>(&args),
        "hex11" => run_net_hex::<11>(&args),
        "chess" => run_net_chess(&args),
        unknown_game => panic!("unknown game: {:?}", unknown_game),
    };
    outputs_to_json(outputs, &args.outfile)
}

fn run_net_tictactoe(args: &Args) -> Vec<(Vec<f32>, f32)> {
    let pos = TttPosition::from_str(&args.position);
    let net = TwoHeadedNetBase::<TttGame>::new(&args.model_path, true, None);
    let samples = (0..args.repeat)
        .map(|_| ttt::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<TttGame>(&samples);
    net.run_net(tensor)
}

fn run_net_hex<const BOARD_SIZE: usize>(args: &Args) -> Vec<(Vec<f32>, f32)> {
    let pos = HexPosition::from_str(&args.position);
    let net = TwoHeadedNetBase::<HexGame<BOARD_SIZE>>::new(&args.model_path, true, None);
    let samples = (0..args.repeat)
        .map(|_| hex::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<HexGame<BOARD_SIZE>>(&samples);
    net.run_net(tensor)
}

fn run_net_chess(args: &Args) -> Vec<(Vec<f32>, f32)> {
    let pos = ChessPosition::from_str(&args.position);

    let net = TwoHeadedNetBase::<ChessGame>::new(&args.model_path, true, None);
    let samples = (0..args.repeat)
        .map(|_| chess::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<ChessGame>(&samples);
    net.run_net(tensor)
}

fn outputs_to_json(outputs: Vec<(Vec<f32>, f32)>, filename: &String) -> std::io::Result<()> {
    let vals = outputs.iter().map(|(_probs, val)| *val).collect_vec();
    let probs = outputs.into_iter().map(|(probs, _val)| probs).collect_vec();
    fs::write(
        filename,
        json::object! {
            probs: probs,
            vals: vals,
        }
        .dump(),
    )
}
