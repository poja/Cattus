use cattus::chess::chess_game::{ChessGame, ChessPosition};
use cattus::game::net;
use cattus::hex::hex_game::{HexGame, HexPosition};
use cattus::ttt::ttt_game::{TttGame, TttPosition};
use cattus::{chess, hex, ttt};
use clap::Parser;
use ndarray::{Array3, Array4, Axis};
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
    outfile: PathBuf,
}

fn main() -> std::io::Result<()> {
    cattus::util::init_globals(None);

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

    // Remove the batch axis
    assert_eq!(tensor.shape()[0], 1);
    let tensor = tensor.remove_axis(Axis(0));

    tensor_to_json(tensor, &args.outfile)
}

fn create_tensor_tictactoe(args: &Args) -> Array4<f32> {
    let pos = TttPosition::from_str(&args.position);
    let planes = ttt::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<TttGame>(&[planes], 1)
}
fn create_tensor_hex<const BOARD_SIZE: usize>(args: &Args) -> Array4<f32> {
    let pos = HexPosition::from_str(&args.position);
    let planes = hex::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<HexGame<BOARD_SIZE>>(&[planes], 1)
}

fn create_tensor_chess(args: &Args) -> Array4<f32> {
    let pos = ChessPosition::from_str(&args.position);
    let planes = chess::net::common::position_to_planes(&pos);
    net::planes_to_tensor::<ChessGame>(&[planes], 1)
}

fn tensor_to_json(tensor: Array3<f32>, filename: &Path) -> std::io::Result<()> {
    fs::write(
        filename,
        json::object! {
            shape: tensor.shape().to_vec(),
            data: tensor.iter().cloned().collect::<Vec<f32>>(),
        }
        .dump(),
    )
}
