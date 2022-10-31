use clap::Parser;
use itertools::Itertools;
use std::fs;
use tensorflow::Tensor;

use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::net;
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
    outfile: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    match args.game.as_str() {
        "tictactoe" => test_tictactoe(args),
        "hex" => test_hex(args),
        "chess" => test_chess(args),
        unknown_game => panic!("unknown game: {:?}", unknown_game),
    }
}

fn test_tictactoe(args: Args) -> std::io::Result<()> {
    let pos = TttPosition::from_str(&args.position);

    let planes = ttt::net::common::position_to_planes(&pos);
    let tensor = net::planes_to_tensor::<TttGame, true>(planes);
    tensor_to_json(tensor, &args.outfile)
}

fn test_hex(args: Args) -> std::io::Result<()> {
    let pos = HexPosition::from_str(&args.position);

    let planes = hex::net::common::position_to_planes(&pos);
    let tensor = net::planes_to_tensor::<HexGame, true>(planes);
    tensor_to_json(tensor, &args.outfile)
}

fn test_chess(args: Args) -> std::io::Result<()> {
    let pos = ChessPosition::from_str(&args.position);

    let planes = chess::net::common::position_to_planes(&pos);
    let tensor = net::planes_to_tensor::<ChessGame, true>(planes);
    tensor_to_json(tensor, &args.outfile)
}

fn tensor_to_json(tensor: Tensor<f32>, filename: &String) -> std::io::Result<()> {
    let shape: Option<Vec<Option<i64>>> = tensor.shape().try_into().unwrap();
    let shape = shape.unwrap().into_iter().map(|d| d.unwrap()).collect_vec();
    let data = tensor.iter().cloned().collect_vec();
    fs::write(
        filename,
        json::object! {
            shape: shape,
            data: data,
        }
        .dump(),
    )
}
