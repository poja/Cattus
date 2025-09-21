use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::model::Model;
use cattus::game::net;
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::{chess, hex, ttt};
use clap::Parser;
use itertools::Itertools;
use ndarray::{Array2, ArrayD, Axis};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    game: String,
    #[clap(long)]
    position: String,
    #[clap(long)]
    model_path: PathBuf,
    #[clap(long)]
    batch_size: usize,
    #[clap(long)]
    outfile: PathBuf,
    #[clap(long, default_value = "1")]
    repeat: u32,
}

fn main() -> std::io::Result<()> {
    cattus::util::init_globals(None);

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
    let mut model = Model::new(&args.model_path);
    let outputs = (0..args.repeat)
        .map(|_| {
            let samples = ttt::net::common::position_to_planes(&pos);
            let tensor = net::planes_to_tensor::<TttGame>(&[samples], args.batch_size);
            model.run(vec![tensor.into_dyn()])
        })
        .collect_vec();
    (0..outputs[0].len())
        .map(|output_idx| {
            let outputs = outputs
                .iter()
                .map(|outputs| outputs[output_idx].view())
                .collect_vec();
            ndarray::concatenate(Axis(0), &outputs).unwrap()
        })
        .collect()
}

fn run_net_hex<const BOARD_SIZE: usize>(args: &Args) -> Vec<ArrayD<f32>> {
    let pos = HexPosition::from_str(&args.position);
    let mut model = Model::new(&args.model_path);
    let outputs = (0..args.repeat)
        .map(|_| {
            let samples = hex::net::common::position_to_planes(&pos);
            let tensor = net::planes_to_tensor::<HexGame<BOARD_SIZE>>(&[samples], args.batch_size);
            model.run(vec![tensor.into_dyn()])
        })
        .collect_vec();
    (0..outputs[0].len())
        .map(|output_idx| {
            let outputs = outputs
                .iter()
                .map(|outputs| outputs[output_idx].view())
                .collect_vec();
            ndarray::concatenate(Axis(0), &outputs).unwrap()
        })
        .collect()
}

fn run_net_chess(args: &Args) -> Vec<ArrayD<f32>> {
    let pos = ChessPosition::from_str(&args.position);
    let mut model = Model::new(&args.model_path);

    let outputs = (0..args.repeat)
        .map(|_| {
            let samples = chess::net::common::position_to_planes(&pos);
            let tensor = net::planes_to_tensor::<ChessGame>(&[samples], args.batch_size);
            model.run(vec![tensor.into_dyn()])
        })
        .collect_vec();
    (0..outputs[0].len())
        .map(|output_idx| {
            let outputs = outputs
                .iter()
                .map(|outputs| outputs[output_idx].view())
                .collect_vec();
            ndarray::concatenate(Axis(0), &outputs).unwrap()
        })
        .collect()
}

fn outputs_to_json(mut outputs: Vec<ArrayD<f32>>, filename: &Path) -> std::io::Result<()> {
    assert_eq!(outputs.len(), 2);
    let probs: Array2<f32> = outputs.remove(0).into_dimensionality().unwrap();
    let vals: Array2<f32> = outputs.remove(0).into_dimensionality().unwrap();
    let probs = probs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect_vec();
    assert_eq!(vals.shape()[1], 1);
    let vals = vals.iter().cloned().collect_vec();

    #[derive(serde::Serialize)]
    struct JsonOutputs {
        probs: Vec<Vec<f32>>,
        vals: Vec<f32>,
    }
    let writer = fs::File::create_new(filename)?;
    serde_json::to_writer(writer, &JsonOutputs { probs, vals })?;
    Ok(())
}
