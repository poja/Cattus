use cattus::game::common::{GameColor, GamePlayer, IGame};
use cattus::hex::hex_game::{HexGameStandard, HexPosition};
use cattus::hex::uxi::HexPlayerUXI;
use cattus::utils;
use clap::Parser;
use rand::Rng;
use std::path::Path;
use std::time::Instant;

fn comapre_engines(
    engine1_filename: &String,
    engine2_filename: &String,
    engine1_params: &Vec<String>,
    engine2_params: &Vec<String>,
    number_of_games: usize,
    _working_dir: &str,
) {
    // let mut rng = rand::thread_rng();
    // let engine1_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    // let engine2_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    let mut engine1 = HexPlayerUXI::new(Path::new(engine1_filename) /*, engine1_errfile*/);
    let mut engine2 = HexPlayerUXI::new(Path::new(engine2_filename) /*, engine2_errfile*/);

    let engine1_started = engine1.start(engine1_params);
    let engine2_started = engine2.start(engine2_params);

    if !engine1_started || !engine2_started {
        if engine1_started {
            engine1.stop();
        }
        if engine2_started {
            engine2.stop();
        }
        log::error!("Failed to start engines.");
        return;
    }

    compare_players(
        &mut engine1,
        &mut engine2,
        number_of_games,
        engine1_filename,
        engine2_filename,
    );

    engine1.stop();
    engine2.stop();
}

fn compare_players(
    player1: &mut dyn GamePlayer<HexGameStandard>,
    player2: &mut dyn GamePlayer<HexGameStandard>,
    number_of_games: usize,
    player1_display_name: &String,
    player2_display_name: &String,
) {
    let mut rng = rand::thread_rng();

    log::info!("Comparing between two players:");
    log::info!("\tplayer1: {}", player1_display_name);
    log::info!("\tplayer2: {}", player2_display_name);
    log::info!("\tnumber of games: {}", number_of_games);

    let run_time = Instant::now();
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    for _ in 0..number_of_games {
        let starting_player = match rng.gen::<bool>() {
            true => GameColor::Player1,
            false => GameColor::Player2,
        };
        let mut game =
            HexGameStandard::new_from_pos(HexPosition::new_with_starting_color(starting_player));
        let (_final_pos, winner) = game.play_until_over(player1, player2);
        if let Some(winner) = winner {
            match winner {
                GameColor::Player1 => player1_wins += 1,
                GameColor::Player2 => player2_wins += 1,
            }
        };
    }
    log::info!("Comparison results:");
    log::info!(
        "\t{}/{} : {}",
        player1_wins,
        number_of_games,
        player1_display_name
    );
    log::info!(
        "\t{}/{} : {}",
        player2_wins,
        number_of_games,
        player2_display_name
    );
    log::info!("\tRunning time: {}s", run_time.elapsed().as_secs());
}

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct Args {
    #[clap(long)]
    engine1: String,
    #[clap(long)]
    engine2: String,
    #[clap(long, default_value = "")]
    engine1_params: String,
    #[clap(long, default_value = "")]
    engine2_params: String,
    #[clap(short, long, default_value_t = 10)]
    repeat: usize,
    #[clap(short, long, default_value = "_CURRENT_DIR_")]
    workdir: String,
}

/**
 * Running example:
 *
 * .\target\debug\hex_uxi_tester.exe
 *      --engine2 .\target\debug\hex_uxi_rand.exe
 *      --engine1 .\target\debug\hex_uxi_mcts.exe --engine1-params " --sim-num 600 --network .\model\m1"
 *
 * Another example:
 * ./target/release/hex_uxi_tester
 *      --engine1 ./target/release/hex_uxi_mcts --engine1-params \"--sim-num\ 600\"
 *      --engine2 ./target/release/hex_uxi_mcts --engine2-params \"--sim-num\ 1500\"
 */
fn main() {
    utils::init_globals();

    let mut args = Args::parse();
    if args.workdir == "_CURRENT_DIR_" {
        args.workdir = String::from(std::env::current_dir().unwrap().to_str().unwrap());
    }
    let parse_engine_args = |engine_args_str0: String| -> Option<Vec<String>> {
        let mut engine_args_str = engine_args_str0;
        if engine_args_str.is_empty() {
            return Some(vec![]);
        }
        if engine_args_str.starts_with('\"') {
            engine_args_str = engine_args_str[1..engine_args_str.len()].to_string();
        }
        if engine_args_str.ends_with('\"') {
            engine_args_str = engine_args_str[0..engine_args_str.len() - 1].to_string();
        }
        Some(
            engine_args_str
                .split(' ')
                .filter(|s| -> bool { !s.is_empty() })
                .map(|s| -> String { String::from(s) })
                .collect(),
        )
    };
    let engine1_params = parse_engine_args(args.engine1_params).unwrap();
    let engine2_params = parse_engine_args(args.engine2_params).unwrap();

    comapre_engines(
        &args.engine1,
        &args.engine2,
        &engine1_params,
        &engine2_params,
        args.repeat,
        &args.workdir,
    );
}
