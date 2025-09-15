use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::{MCTSPlayer, ValueFunction};
use crate::game::self_play::{DataSerializer, SelfPlayRunner};
use crate::game::utils::Callback;
use crate::util::{self, Builder, Device};
use clap::Parser;
use itertools::Itertools;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

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

#[derive(Copy, Clone)]
struct MetricsData {
    pub search_count: u32,
    pub search_duration: Duration,
}
impl MetricsData {
    fn get_search_duration_average(&self) -> Duration {
        if self.search_count > 0 {
            self.search_duration / self.search_count
        } else {
            Duration::ZERO
        }
    }
}

struct Metrics {
    data: Mutex<MetricsData>,
}
impl Metrics {
    fn new() -> Self {
        Self {
            data: Mutex::new(MetricsData {
                search_count: 0,
                search_duration: Duration::ZERO,
            }),
        }
    }

    fn add_search_duration(&self, duration: Duration) {
        let mut data = self.data.lock().unwrap();
        data.search_count += 1;
        data.search_duration += duration;
    }

    fn get_raw_data(&self) -> MetricsData {
        *self.data.lock().unwrap().deref()
    }
}

struct SearchDurationCallback {
    metrics: Arc<Metrics>,
}
impl Callback<Duration> for SearchDurationCallback {
    fn call(&self, duration: Duration) {
        self.metrics.add_search_duration(duration)
    }
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
    metrics: Arc<Metrics>,
}

impl<Game: IGame> PlayerBuilder<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        network: Arc<dyn ValueFunction<Game>>,
        sim_num: u32,
        explore_factor: f32,
        prior_noise_alpha: f32,
        prior_noise_epsilon: f32,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            network,
            sim_num,
            explore_factor,
            prior_noise_alpha,
            prior_noise_epsilon,
            metrics,
        }
    }
}

impl<Game: IGame> Builder<MCTSPlayer<Game>> for PlayerBuilder<Game> {
    fn build(&self) -> MCTSPlayer<Game> {
        let mut player = MCTSPlayer::new_custom(
            self.sim_num,
            self.explore_factor,
            self.prior_noise_alpha,
            self.prior_noise_epsilon,
            self.network.clone(),
        );

        player.set_search_duration_callback(Some(Box::new(SearchDurationCallback {
            metrics: self.metrics.clone(),
        })));

        player
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

    fn get_hits_counter(&self) -> usize {
        self.caches
            .iter()
            .map(|cache| cache.get_hits_counter())
            .sum()
    }

    fn get_misses_counter(&self) -> usize {
        self.caches
            .iter()
            .map(|cache| cache.get_misses_counter())
            .sum()
    }
}

pub fn run_main<Game: IGame + 'static>(
    network_builder: Box<dyn INNetworkBuilder<Game>>,
    serializer: Box<dyn DataSerializer<Game>>,
) -> std::io::Result<()> {
    util::init_globals();
    let args = SelfPlayArgs::parse();
    let device = match args.device.to_uppercase().as_str() {
        "CPU" => Device::Cpu,
        "GPU" => Device::Cuda,
        "MPS" => Device::Mps,
        unknown_pu => panic!("unknown processing unit '{unknown_pu}'"),
    };

    let metrics = Arc::new(Metrics::new());
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
        metrics.clone(),
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
            metrics.clone(),
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
        let cache_hits = cache_builder.get_hits_counter();
        let cache_uses = cache_hits + cache_builder.get_misses_counter();

        let nets_stats = nets.iter().map(|net| net.get_statistics()).collect_vec();
        let net_activation_count: usize = nets_stats
            .iter()
            .map(|net_stats| net_stats.activation_count.unwrap())
            .sum();
        let net_run_duration_average: Duration = nets_stats
            .iter()
            .map(|net_stats| {
                net_stats.run_duration_average.unwrap() * net_stats.activation_count.unwrap() as u32
            })
            .sum::<Duration>()
            / net_activation_count as u32;
        let batch_size_average: f32 = nets_stats
            .iter()
            .map(|net_stats| {
                net_stats.batch_size_average.unwrap() * net_stats.activation_count.unwrap() as f32
            })
            .sum::<f32>()
            / net_activation_count as f32;

        let metrics = metrics.get_raw_data();

        fs::write(
            &summary_file,
            json::object! {
                player1_wins: result.w1,
                player2_wins: result.w2,
                draws: result.d,
                net_activations_count: net_activation_count,
                net_run_duration_average_us: net_run_duration_average.as_micros() as u64,
                batch_size_average: batch_size_average,
                search_count: metrics.search_count,
                search_duration_average_ms: metrics.get_search_duration_average().as_millis() as u64,
                cache_hit_ratio: cache_hits as f32 / cache_uses as f32,
            }
            .dump(),
        )?;
    }

    Ok(())
}
