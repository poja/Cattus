use clap::Parser;
use itertools::Itertools;
use std::fs;

use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::net::{self, TwoHeadedNetBase};
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::{chess, hex, ttt};

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
    let args = Args::parse();

    let outputs = match args.game.as_str() {
        "tictactoe" => run_net_tictactoe(&args),
        "hex5" => run_net_hex::<5>(&args),
        "hex7" => run_net_hex::<7>(&args),
        "hex9" => run_net_hex::<9>(&args),
        "hex11" => run_net_hex::<11>(&args),
        "chess" => run_net_chess(&args),
        unknown_game => panic!("unknown game: {:?}", unknown_game),
    };
    outputs_to_json(outputs, &args.outfile)
}

fn run_net_tictactoe(args: &Args) -> Vec<(f32, Vec<f32>)> {
    let pos = TttPosition::from_str(&args.position);

    let net = TwoHeadedNetBase::<TttGame, true>::new(&args.model_path, None);
    let mut outputs = Vec::new();
    for _ in 0..args.repeat {
        let planes = ttt::net::common::position_to_planes(&pos);
        let tensor = net::planes_to_tensor::<TttGame, true>(planes);
        outputs.push(net.run_net(tensor));
    }
    outputs
}

fn run_net_hex<const BOARD_SIZE: usize>(args: &Args) -> Vec<(f32, Vec<f32>)> {
    let mut outputs = Vec::new();
    let pos = HexPosition::from_str(&args.position);
    let net = TwoHeadedNetBase::<HexGame<BOARD_SIZE>, true>::new(&args.model_path, None);
    for _ in 0..args.repeat {
        let planes = hex::net::common::position_to_planes(&pos);
        let tensor = net::planes_to_tensor::<HexGame<BOARD_SIZE>, true>(planes);
        outputs.push(net.run_net(tensor));
    }
    outputs
}

fn run_net_chess(args: &Args) -> Vec<(f32, Vec<f32>)> {
    let pos = ChessPosition::from_str(&args.position);

    let net = TwoHeadedNetBase::<ChessGame, true>::new(&args.model_path, None);
    let mut outputs = Vec::new();
    for _ in 0..args.repeat {
        let planes = chess::net::common::position_to_planes(&pos);
        let tensor = net::planes_to_tensor::<ChessGame, true>(planes);
        outputs.push(net.run_net(tensor));
    }
    outputs
}

fn outputs_to_json(outputs: Vec<(f32, Vec<f32>)>, filename: &String) -> std::io::Result<()> {
    let vals = outputs.iter().map(|(val, _probs)| *val).collect_vec();
    let probs = outputs.into_iter().map(|(_val, probs)| probs).collect_vec();
    fs::write(
        filename,
        json::object! {
            vals: vals,
            probs: probs,
        }
        .dump(),
    )
}
