use clap::Parser;
use hex_backend::uxi::UXI;
use std::path::Path;

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

    UXI::compare_engines(
        &Path::new(&args.engine1),
        &Path::new(&args.engine2),
        args.repeat,
        &Path::new(&args.workdir),
    );
}
