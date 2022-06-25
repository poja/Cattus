// use clap::Parser;
use rl::game_utils::{mcts, game};
use rl::hex::hex_game;
use rl::game_utils::train;

// #[derive(Parser, Debug)]
// #[clap(about, long_about = None)]
// struct Args {
//     #[clap(long, default_value = "100")]
//     sim_count: u32,
//     #[clap(long, default_value = "1.41421")]
//     explore_param_c: f32,
// }


struct MyEncode {
}

impl MyEncode {

pub fn new() -> Self {
    Self {
    }
}
}

impl train::Encoder<hex_game::HexGame> for MyEncode {
    fn encode_moves(&self, moves: &Vec<(hex_game::Location, f32)>) -> Vec<f32> {
        return vec![];
    }
    fn decode_moves(&self, moves: &Vec<f32>) -> Vec<(hex_game::Location, f32)>{
        return vec![];
    }
    fn encode_position(&self, position: &hex_game::HexPosition) -> Vec<f32>{
        let mut vec = Vec::new();
        for r in 0..hex_game::BOARD_SIZE {
            for c in 0..hex_game::BOARD_SIZE {
                vec.push(match position.get_tile(r, c) {
                    hex_game::Hexagon::Full(color) => {
                        match color {
                            game::GameColor::Player1 => 1.0,
                            game::GameColor::Player2 => -1.0,
                        }
                    }
                    hex_game::Hexagon::Empty => 0.0
                });
            }
        }

        return vec;
    }
}


fn main() {
    // let args = Args::parse();
    let mut encoder = MyEncode::new();
    let trainer = train::Trainer::new(&mut encoder);
    let mut player = mcts::MCTSPlayer::new_custom(100, 1.4);
    let out_path = String::from("C:/code/rl/out");

    trainer.generate_data(&mut player, 1, &out_path);
}
