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
    let tensor = match args.game.as_str() {
        "tictactoe" => create_tensor_tictactoe(&args),
        "hex4" => create_tensor_hex::<4>(&args),
        "hex5" => create_tensor_hex::<5>(&args),
        "hex7" => create_tensor_hex::<7>(&args),
        "hex9" => create_tensor_hex::<9>(&args),
        "hex11" => create_tensor_hex::<11>(&args),
        "chess" => create_tensor_chess(&args),
        unknown_game => panic!("unknown game: {:?}", unknown_game),
    };
    tensor_to_json(tensor, &args.outfile)
}

fn create_tensor_tictactoe(args: &Args) -> Tensor<f32> {
    let pos = TttPosition::from_str(&args.position);
    let planes = ttt::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<TttGame, true>(planes)
}
fn create_tensor_hex<const BOARD_SIZE: usize>(args: &Args) -> Tensor<f32> {
    let pos = HexPosition::from_str(&args.position);
    let planes = hex::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<HexGame<BOARD_SIZE>, true>(planes)
}

fn create_tensor_chess(args: &Args) -> Tensor<f32> {
    let pos = ChessPosition::from_str(&args.position);
    let planes = chess::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<ChessGame, true>(planes)
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
