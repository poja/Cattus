use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::model::Model;
use cattus::game::net;
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::utils;
use cattus::{chess, hex, ttt};
use clap::Parser;
use itertools::Itertools;
use ndarray::{Array2, ArrayD};
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
    utils::init_globals();

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

fn run_net_tictactoe(args: &Args) -> Vec<ArrayD<f32>> {
    let pos = TttPosition::from_str(&args.position);
    let model = Model::new(&args.model_path);
    let samples = (0..args.repeat)
        .map(|_| ttt::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<TttGame>(&samples);
    model.run(vec![tensor.into_dyn()])
}

fn run_net_hex<const BOARD_SIZE: usize>(args: &Args) -> Vec<ArrayD<f32>> {
    let pos = HexPosition::from_str(&args.position);
    let model = Model::new(&args.model_path);
    let samples = (0..args.repeat)
        .map(|_| hex::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<HexGame<BOARD_SIZE>>(&samples);
    model.run(vec![tensor.into_dyn()])
}

fn run_net_chess(args: &Args) -> Vec<ArrayD<f32>> {
    let pos = ChessPosition::from_str(&args.position);
    let model = Model::new(&args.model_path);
    let samples = (0..args.repeat)
        .map(|_| chess::net::common::position_to_planes(&pos))
        .collect_vec();
    let tensor = net::planes_to_tensor::<ChessGame>(&samples);
    model.run(vec![tensor.into_dyn()])
}

fn outputs_to_json(mut outputs: Vec<ArrayD<f32>>, filename: &String) -> std::io::Result<()> {
    assert_eq!(outputs.len(), 2);
    let probs: Array2<f32> = outputs.remove(0).into_dimensionality().unwrap();
    let vals: Array2<f32> = outputs.remove(0).into_dimensionality().unwrap();
    let probs = probs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect_vec();
    assert_eq!(vals.shape()[1], 1);
    let vals = vals.into_raw_vec();
    fs::write(
        filename,
        json::object! {
            probs: probs,
            vals: vals,
        }
        .dump(),
    )
}
