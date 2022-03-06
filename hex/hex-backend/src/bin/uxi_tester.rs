use clap::Parser;
use hex_backend::hex_game::{Color, HexGame, HexPlayer};
use hex_backend::uxi::HexPlayerUXI;
use rand::Rng;
use std::path::Path;
use std::time::{Duration, Instant};

fn comapre_engines(
    engine1_filename: &String,
    engine2_filename: &String,
    number_of_games: usize,
    working_dir: &String,
) {
    // let mut rng = rand::thread_rng();
    // let engine1_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    // let engine2_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
    let mut engine1 = HexPlayerUXI::new(&Path::new(engine1_filename) /*, engine1_errfile*/);
    let mut engine2 = HexPlayerUXI::new(&Path::new(engine2_filename) /*, engine2_errfile*/);

    let engine1_started = engine1.start();
    let engine2_started = engine2.start();

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
    player1: &mut dyn HexPlayer,
    player2: &mut dyn HexPlayer,
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
            true => Color::Red,
            false => Color::Blue,
        };
        let mut game = HexGame::new(starting_player, player1, player2);
        match game.play_until_over() {
            None => {}
            Some(winner) => match winner {
                Color::Red => player1_wins += 1,
                Color::Blue => player2_wins += 1,
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
 */

fn main() {
    let mut args = Args::parse();
    if args.workdir == "_CURRENT_DIR_" {
        args.workdir = String::from(std::env::current_dir().unwrap().to_str().unwrap());
    }

    comapre_engines(&args.engine1, &args.engine2, args.repeat, &args.workdir);
}
