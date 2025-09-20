use cattus::game::cache::ValueFuncCache;
use cattus::game::common::IGame;
use cattus::game::mcts::{MctsPlayer, ValueFunction};
use cattus::util::{self, Builder, Device};
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::self_play::{GameExt, SelfPlayRunner};
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

struct PlayerBuilder<Game: IGame> {
    network: Arc<dyn ValueFunction<Game>>,
    sim_num: u32,
    explore_factor: f32,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
}

impl<Game: IGame> PlayerBuilder<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        network: Arc<dyn ValueFunction<Game>>,
        sim_num: u32,
        explore_factor: f32,
        prior_noise_alpha: f32,
        prior_noise_epsilon: f32,
    ) -> Self {
        Self {
            network,
            sim_num,
            explore_factor,
            prior_noise_alpha,
            prior_noise_epsilon,
        }
    }
}

impl<Game: IGame> Builder<MctsPlayer<Game>> for PlayerBuilder<Game> {
    fn build(&self) -> MctsPlayer<Game> {
        MctsPlayer::new_custom(
            self.sim_num,
            self.explore_factor,
            self.prior_noise_alpha,
            self.prior_noise_epsilon,
            self.network.clone(),
        )
    }
}

struct CacheBuilder<Game: IGame> {
    max_size: usize,
    caches: Vec<Arc<ValueFuncCache<Game>>>,
}
impl<Game: IGame> CacheBuilder<Game> {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            caches: vec![],
        }
    }
    fn build_cache(&mut self) -> Arc<ValueFuncCache<Game>> {
        let cache = Arc::new(ValueFuncCache::new(self.max_size));
        self.caches.push(cache.clone());
        cache
    }
}

pub fn run_main<Game: GameExt + 'static>(
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

    let mut cache_builder = CacheBuilder::new(args.cache_size);
    let mut nets = vec![];

    let player1_net: Arc<dyn ValueFunction<Game>> = Arc::from(network_builder.build_net(
        &args.model1_path,
        cache_builder.build_cache(),
        device,
        args.batch_size,
    ));
    nets.push(player1_net.clone());
    let player1_builder = Arc::new(PlayerBuilder::new(
        player1_net,
        args.sim_num,
        args.explore_factor,
        args.prior_noise_alpha,
        args.prior_noise_epsilon,
    ));

    let player2_builder = if args.model1_path == args.model2_path {
        player1_builder.clone()
    } else {
        let player2_net: Arc<dyn ValueFunction<Game>> = Arc::from(network_builder.build_net(
            &args.model2_path,
            cache_builder.build_cache(),
            device,
            args.batch_size,
        ));
        nets.push(player2_net.clone());
        Arc::new(PlayerBuilder::new(
            player2_net,
            args.sim_num,
            args.explore_factor,
            args.prior_noise_alpha,
            args.prior_noise_epsilon,
        ))
    };

    let result = SelfPlayRunner::new(
        player1_builder,
        player2_builder,
        args.temperature_policy,
        Arc::from(serializer),
        args.threads,
    )
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
                        .map(|v| {
                            serde_json::Value::Number(serde_json::Number::from_f64(v.0).unwrap())
                        })
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
