use clap::Parser;
use rand::Rng;
use rl::game_utils::game::{GameColor, GamePlayer, IGame};
use rl::hex::hex_game::{HexGame, HexPosition};
use rl::hex::uxi::HexPlayerUXI;
use std::path::Path;
use std::time::Instant;

fn comapre_engines(
    engine1_filename: &String,
    engine2_filename: &String,
    engine1_params: &Vec<String>,
    engine2_params: &Vec<String>,
    number_of_games: usize,
    _working_dir: &String,
) {
    // let mut rng = rand::thread_rng();
    // let engine1_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    // let engine2_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    let mut engine1 = HexPlayerUXI::new(&Path::new(engine1_filename) /*, engine1_errfile*/);
    let mut engine2 = HexPlayerUXI::new(&Path::new(engine2_filename) /*, engine2_errfile*/);

    let engine1_started = engine1.start(engine1_params);
    let engine2_started = engine2.start(engine2_params);

    if !engine1_started || !engine2_started {
        if engine1_started {
            engine1.stop();
        }
        if engine2_started {
            engine2.stop();
        }
        eprintln!("Failed to start engines.");
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
    player1: &mut dyn GamePlayer<HexGame>,
    player2: &mut dyn GamePlayer<HexGame>,
    number_of_games: usize,
    player1_display_name: &String,
    player2_display_name: &String,
) {
    let mut rng = rand::thread_rng();

    println!("Comparing between two players:");
    println!("\tplayer1: {}", player1_display_name);
    println!("\tplayer2: {}", player2_display_name);
    println!("\tnumber of games: {}", number_of_games);

    // TODO progress bar
    let run_time = Instant::now();
    let mut player1_wins = 0;
    let mut player2_wins = 0;
    for _ in 0..number_of_games {
        let starting_player = match rng.gen::<bool>() {
            true => GameColor::Player1,
            false => GameColor::Player2,
        };
        match HexGame::play_until_over(
            &HexPosition::new_with_starting_color(starting_player),
            player1,
            player2,
        )
        .1
        {
            None => {}
            Some(winner) => match winner {
                GameColor::Player1 => player1_wins += 1,
                GameColor::Player2 => player2_wins += 1,
            },
        };
    }
    println!("Comparison results:");
    println!(
        "\t{}/{} : {}",
        player1_wins, number_of_games, player1_display_name
    );
    println!(
        "\t{}/{} : {}",
        player2_wins, number_of_games, player2_display_name
    );
    println!("\tRunning time: {}s", run_time.elapsed().as_secs());
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
 * .\target\debug\uxi_tester.exe
 *      --engine1 .\target\debug\uxi_mcts.exe
 *      --engine2 .\target\debug\uxi_rand.exe
 *
 * Another example:
 * ./target/release/uxi_tester
 *      --engine1 ./target/release/uxi_mcts --engine1-params \"--sim-count\ 600\ --explore-param-c\ 10\"
 *      --engine2 ./target/release/uxi_mcts --engine2-params \"--sim-count\ 1500\"
 */

fn main() {
    let mut args = Args::parse();
    if args.workdir == "_CURRENT_DIR_" {
        args.workdir = String::from(std::env::current_dir().unwrap().to_str().unwrap());
    }
    let parse_engine_args = |engine_args_str: &String| -> Option<Vec<String>> {
        if engine_args_str.len() == 0 {
            return Some(vec![]);
        }
        if !(engine_args_str.len() >= 2
            && engine_args_str.starts_with("\"")
            && engine_args_str.ends_with("\""))
        {
            eprintln!("Engine args must be wrapper with \"_args_\"");
            return None;
        }
        Some(
            engine_args_str[1..engine_args_str.len() - 1]
                .split(" ")
                .map(|s| -> String { String::from(s) })
                .collect(),
        )
    };
    let engine1_params = parse_engine_args(&args.engine1_params).unwrap();
    let engine2_params = parse_engine_args(&args.engine2_params).unwrap();

    comapre_engines(
        &args.engine1,
        &args.engine2,
        &engine1_params,
        &engine2_params,
        args.repeat,
        &args.workdir,
    );
}
