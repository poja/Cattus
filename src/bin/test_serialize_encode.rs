use std::fs;

use clap::Parser;
use itertools::Itertools;
use rl::chess::chess_game::{ChessGame, ChessPosition};
use rl::chess::net::serializer::ChessSerializer;
use rl::game::common::{GameColor, GamePosition, IGame};
use rl::game::net;
use rl::game::self_play::DataSerializer;
use rl::hex::hex_game::{HexGame, HexPosition};
use rl::hex::net::serializer::HexSerializer;
use rl::ttt::net::serializer::TttSerializer;
use rl::ttt::ttt_game::{TttGame, TttPosition};
use rl::{chess, hex, ttt};
use tensorflow::Tensor;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    game: String,
    #[clap(long)]
    position: String,
    #[clap(long)]
    serialize_out: String,
    #[clap(long)]
    encode_out: String,
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
    let tensor = net::planes_to_tensor::<TttGame>(planes);
    tensor_to_json(tensor, &args.encode_out)?;

    let serializer = TttSerializer {};
    serialize_position(pos, &serializer, &args.serialize_out)
}

fn test_hex(args: Args) -> std::io::Result<()> {
    let pos = HexPosition::from_str(&args.position);

    let planes = hex::net::common::position_to_planes(&pos);
    let tensor = net::planes_to_tensor::<HexGame>(planes);
    tensor_to_json(tensor, &args.encode_out)?;

    let serializer = HexSerializer {};
    serialize_position(pos, &serializer, &args.serialize_out)
}

fn test_chess(args: Args) -> std::io::Result<()> {
    let pos = ChessPosition::from_str(&args.position);

    let planes = chess::net::common::position_to_planes(&pos);
    let tensor = net::planes_to_tensor::<ChessGame>(planes);
    tensor_to_json(tensor, &args.encode_out)?;

    let serializer = ChessSerializer {};
    serialize_position(pos, &serializer, &args.serialize_out)
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

fn serialize_position<Game: IGame>(
    pos: Game::Position,
    serializer: &dyn DataSerializer<Game>,
    filename: &str,
) -> std::io::Result<()> {
    let moves = pos.get_legal_moves();
    let moves_num = moves.len();
    let probs = moves
        .into_iter()
        .enumerate()
        .map(|(idx, m)| (m, idx as f32 / (moves_num * (moves_num - 1)) as f32))
        .collect_vec();
    let winner = match moves_num % 3 {
        0 => Some(GameColor::Player1),
        1 => Some(GameColor::Player2),
        2 => None,
        _ => panic!("cant happen"),
    };
    serializer.serialize_data_entry(pos, probs, winner, filename)
}
