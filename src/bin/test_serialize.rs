use cattus::chess::chess_game::ChessPosition;
use cattus::chess::net::serializer::ChessSerializer;
use cattus::game::common::{GameColor, GamePosition, IGame};
use cattus::game::self_play::DataSerializer;
use cattus::hex::hex_game::HexPosition;
use cattus::hex::net::serializer::HexSerializer;
use cattus::ttt::net::serializer::TttSerializer;
use cattus::ttt::ttt_game::TttPosition;
use clap::Parser;
use itertools::Itertools;

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

    let serializer = TttSerializer {};
    serialize_position(pos, &serializer, &args.outfile)
}

fn test_hex(args: Args) -> std::io::Result<()> {
    let pos = HexPosition::from_str(&args.position);

    let serializer = HexSerializer {};
    serialize_position(pos, &serializer, &args.outfile)
}

fn test_chess(args: Args) -> std::io::Result<()> {
    let pos = ChessPosition::from_str(&args.position);

    let serializer = ChessSerializer {};
    serialize_position(pos, &serializer, &args.outfile)
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