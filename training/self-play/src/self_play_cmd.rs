use cattus::mcts::cache::ValueFuncCache;
use cattus::mcts::value_func::ValueFunction;
use cattus::mcts::{MctsParams, TemperaturePolicy};
use cattus::net::model::InferenceConfig;
use cattus::net::NNetwork;
use cattus::util;
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
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
    #[clap(long)]
    config_file: PathBuf,
}

#[derive(serde::Deserialize)]
struct Config {
    model: ModelConfig,
    mcts: MctsConfig,
    threads: u32,
}
#[derive(serde::Deserialize)]
struct ModelConfig {
    inference: InferenceConfig,
    batch_size: usize,
}
#[derive(serde::Deserialize)]
struct MctsConfig {
    sim_num: u32,
    explore_factor: f32,
    temperature_policy: Vec<(usize, f32)>,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    cache_size: usize,
}

pub fn run_main<Game>(serializer: Box<dyn DataSerializer<Game>>) -> std::io::Result<()>
where
    Game: cattus::game::Game + 'static,
    NNetwork<Game>: ValueFunction<Game>,
{
    util::init_globals();
    let args = SelfPlayArgs::parse();

    let metrics_snapshotter = args.summary_file.is_some().then(|| {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().unwrap();
        snapshotter
    });

    let config: Config =
        serde_json::from_reader(std::fs::File::open(&args.config_file)?).expect("failed to read config file");

    assert!(!config.mcts.temperature_policy.is_empty());
    let scheduled_temperatures = &config.mcts.temperature_policy[..config.mcts.temperature_policy.len() - 1];
    let last_temperature = config.mcts.temperature_policy.last().unwrap().1;
    let temperature = TemperaturePolicy::scheduled(scheduled_temperatures.to_vec(), last_temperature);

    let player1_net: Arc<dyn ValueFunction<Game>> = Arc::new(NNetwork::new(
        &args.model1_path,
        config.model.inference,
        config.model.batch_size,
        Some(Arc::new(ValueFuncCache::new(config.mcts.cache_size))),
    ));
    let player1_params = MctsParams {
        sim_num: config.mcts.sim_num,
        explore_factor: config.mcts.explore_factor,
        temperature: temperature.clone(),
        prior_noise_alpha: config.mcts.prior_noise_alpha,
        prior_noise_epsilon: config.mcts.prior_noise_epsilon,
        value_func: player1_net,
    };

    let player2_params = if args.model1_path == args.model2_path {
        player1_params.clone()
    } else {
        let player2_net: Arc<dyn ValueFunction<Game>> = Arc::new(NNetwork::new(
            &args.model2_path,
            config.model.inference,
            config.model.batch_size,
            Some(Arc::new(ValueFuncCache::new(config.mcts.cache_size))),
        ));
        MctsParams {
            value_func: player2_net,
            ..player1_params.clone()
        }
    };

    let result = SelfPlayRunner::new(player1_params, player2_params, Arc::from(serializer), config.threads)
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
