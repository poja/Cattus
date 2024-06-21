use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::net;
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::utils;
use cattus::{chess, hex, ttt};
use clap::Parser;
use ndarray::Array4;
use std::fs;

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
    utils::init_python();

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

fn create_tensor_tictactoe(args: &Args) -> Array4<f32> {
    let pos = TttPosition::from_str(&args.position);
    let planes = ttt::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<TttGame, true>(&[planes])
}
fn create_tensor_hex<const BOARD_SIZE: usize>(args: &Args) -> Array4<f32> {
    let pos = HexPosition::from_str(&args.position);
    let planes = hex::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<HexGame<BOARD_SIZE>, true>(&[planes])
}

fn create_tensor_chess(args: &Args) -> Array4<f32> {
    let pos = ChessPosition::from_str(&args.position);
    let planes = chess::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<ChessGame, true>(&[planes])
}

fn tensor_to_json(tensor: Array4<f32>, filename: &String) -> std::io::Result<()> {
    fs::write(
        filename,
        json::object! {
            shape: tensor.shape().to_vec(),
            data: tensor.into_raw_vec(),
        }
        .dump(),
    )
}
