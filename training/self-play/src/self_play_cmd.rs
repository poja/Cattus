use cattus::game::cache::ValueFuncCache;
use cattus::game::common::IGame;
use cattus::game::mcts::{MctsParams, TemperaturePolicy, ValueFunction};
use cattus::util::{self, Device};
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::self_play::SelfPlayRunner;
use crate::serialize::DataSerializer;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct SelfPlayArgs {
    #[clap(long)]
    model1_path: PathBuf,
    #[clap(long)]
    model2_path: PathBuf,
    #[clap(long)]
    games_num: u32,
    #[clap(long)]
    out_dir1: PathBuf,
    #[clap(long)]
    out_dir2: PathBuf,
    #[clap(long)]
    summary_file: Option<PathBuf>,
    #[clap(long, default_value = "100")]
    sim_num: u32,
    #[clap(long)]
    batch_size: usize,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
    #[clap(long, default_value = "1.0")]
    temperature_policy: String,
    #[clap(long, default_value = "0.0")]
    prior_noise_alpha: f32,
    #[clap(long, default_value = "0.0")]
    prior_noise_epsilon: f32,
    #[clap(long, default_value = "1")]
    threads: u32,
    #[clap(long, default_value = "CPU")]
    device: String,
    #[clap(long, default_value = "100000")]
    cache_size: usize,
}

pub trait INNetworkBuilder<Game: IGame>: Sync + Send {
    fn build_net(
        &self,
        model_path: &Path,
        cache: Arc<ValueFuncCache<Game>>,
        device: Device,
        batch_size: usize,
    ) -> Box<dyn ValueFunction<Game>>;
}

pub fn run_main<Game: IGame + 'static>(
    network_builder: Box<dyn INNetworkBuilder<Game>>,
    serializer: Box<dyn DataSerializer<Game>>,
) -> std::io::Result<()> {
    util::init_globals(None);
    let args = SelfPlayArgs::parse();

    let metrics_snapshotter = args.summary_file.is_some().then(|| {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().unwrap();
        snapshotter
    });

    let device = match args.device.to_uppercase().as_str() {
        "CPU" => Device::Cpu,
        "GPU" => Device::Cuda,
        "MPS" => Device::Mps,
        unknown_pu => panic!("unknown processing unit '{unknown_pu}'"),
    };

    let temperature = temperature_from_str(&args.temperature_policy);
    let player1_net: Arc<dyn ValueFunction<Game>> = Arc::from(network_builder.build_net(
        &args.model1_path,
        Arc::new(ValueFuncCache::new(args.cache_size)),
        device,
        args.batch_size,
    ));
    let player1_params = MctsParams {
        sim_num: args.sim_num,
        explore_factor: args.explore_factor,
        temperature: temperature.clone(),
        prior_noise_alpha: args.prior_noise_alpha,
        prior_noise_epsilon: args.prior_noise_epsilon,
        value_func: player1_net,
    };

    let player2_params = if args.model1_path == args.model2_path {
        player1_params.clone()
    } else {
        let player2_net: Arc<dyn ValueFunction<Game>> = Arc::from(network_builder.build_net(
            &args.model2_path,
            Arc::new(ValueFuncCache::new(args.cache_size)),
            device,
            args.batch_size,
        ));
        MctsParams {
            sim_num: args.sim_num,
            explore_factor: args.explore_factor,
            temperature: temperature.clone(),
            prior_noise_alpha: args.prior_noise_alpha,
            prior_noise_epsilon: args.prior_noise_epsilon,
            value_func: player2_net,
        }
    };

    let result = SelfPlayRunner::new(player1_params, player2_params, Arc::from(serializer), args.threads)
        .generate_data(args.games_num as usize, &args.out_dir1, &args.out_dir2)?;

    if let Some(summary_file) = args.summary_file {
        let mut metrics = HashMap::new();
        for (key, _unit, _desc, value) in metrics_snapshotter.unwrap().snapshot().into_vec() {
            let key = key.key().name().to_string();
            let value = match value {
                metrics_util::debugging::DebugValue::Counter(value) => {
                    serde_json::Value::Number(serde_json::Number::from(value))
                }
                metrics_util::debugging::DebugValue::Gauge(value) => {
                    serde_json::Value::Number(serde_json::Number::from_f64(value.0).unwrap())
                }
                metrics_util::debugging::DebugValue::Histogram(values) => serde_json::Value::Array(
                    values
                        .into_iter()
                        .map(|v| serde_json::Value::Number(serde_json::Number::from_f64(v.0).unwrap()))
                        .collect(),
                ),
            };
            if metrics.contains_key(&key) {
                panic!()
            }
            metrics.insert(key, value);
        }

        #[derive(serde::Serialize)]
        struct Summary {
            player1_wins: u32,
            player2_wins: u32,
            draws: u32,
            metrics: HashMap<String, serde_json::Value>,
        }
        let summary = Summary {
            player1_wins: result.w1,
            player2_wins: result.w2,
            draws: result.d,
            metrics,
        };
        let writer = std::fs::File::create_new(summary_file).unwrap();
        serde_json::to_writer(writer, &summary).unwrap()
    }

    Ok(())
}

/// Create a scheduler from a string describing the temperature policy
///
/// # Arguments
///
/// * `s` - A string representing the temperature policy. The string should contain an odd number of numbers,
///   with a ',' between them.
///
/// The string will be split into pairs of two numbers, and a final number.
/// Each pair should be of the form (moves_num, temperature) and the final number is the final temperature.
/// Each pair represent an interval of moves number in which the corresponding temperature will be assigned.
/// The pairs should be ordered by the moves_num.
///
/// # Examples
///
/// "1.0" means a constant temperature of 1
/// "30,1.0,0.0" means a temperature of 1.0 for the first 30 moves, and temperature of zero after than
/// "15,2.0,30,0.5,0.1" means a temperature of 2.0 for the first 15 moves, 0.5 in the moves 16 up to 30, and 0.1
/// after that
pub fn temperature_from_str(s: &str) -> TemperaturePolicy {
    let s = s.split(',').collect::<Vec<_>>();
    assert!(s.len() % 2 == 1);

    let mut temperatures = Vec::new();
    for i in 0..((s.len() - 1) / 2) {
        let threshold = s[i * 2].parse::<u32>().unwrap();
        let temperature = s[i * 2 + 1].parse::<f32>().unwrap();
        if !temperatures.is_empty() {
            let (last_threshold, _last_temp) = temperatures.last().unwrap();
            assert!(*last_threshold < threshold);
        }
        temperatures.push((threshold, temperature));
    }
    let last_temp = s.last().unwrap().parse::<f32>().unwrap();
    TemperaturePolicy::scheduled(temperatures, last_temp)
}
