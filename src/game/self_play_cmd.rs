use clap::Parser;
use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crate::game::cache::ValueFuncCache;
use crate::game::common::IGame;
use crate::game::mcts::{MCTSPlayer, ValueFunction};
use crate::game::self_play::{DataSerializer, SelfPlayRunner};
use crate::game::utils::Callback;
use crate::utils::Builder;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct SelfPlayArgs {
    #[clap(long)]
    model1_path: String,
    #[clap(long)]
    model2_path: String,
    #[clap(long)]
    games_num: u32,
    #[clap(long)]
    out_dir1: String,
    #[clap(long)]
    out_dir2: String,
    #[clap(long, default_value = "_NONE_")]
    summary_file: String,
    #[clap(long, default_value = "100")]
    sim_num: u32,
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
    processing_unit: String,
    #[clap(long, default_value = "100000")]
    cache_size: usize,
}

struct MetricsInner {
    net_run_count: u32,
    net_run_duration: Duration,
    search_count: u32,
    search_duration: Duration,
}
struct Metrics {
    data: Mutex<MetricsInner>,
}
impl Metrics {
    fn new() -> Self {
        Self {
            data: Mutex::new(MetricsInner {
                net_run_count: 0,
                net_run_duration: Duration::ZERO,
                search_count: 0,
                search_duration: Duration::ZERO,
            }),
        }
    }

    fn add_net_run_duration(&self, duration: Duration) {
        let mut data = self.data.lock().unwrap();
        data.net_run_count += 1;
        data.net_run_duration += duration;
    }

    fn get_net_run_duration_average(&self) -> Duration {
        let data = self.data.lock().unwrap();
        if data.net_run_count > 0 {
            data.net_run_duration / data.net_run_count
        } else {
            Duration::ZERO
        }
    }

    fn add_search_duration(&self, duration: Duration) {
        let mut data = self.data.lock().unwrap();
        data.search_count += 1;
        data.search_duration += duration;
    }

    fn get_search_duration_average(&self) -> Duration {
        let data = self.data.lock().unwrap();
        if data.search_count > 0 {
            data.search_duration / data.search_count
        } else {
            Duration::ZERO
        }
    }
}

struct NetRunDurationCallback {
    metrics: Arc<Metrics>,
}
impl Callback<Duration> for NetRunDurationCallback {
    fn call(&self, duration: Duration) {
        self.metrics.add_net_run_duration(duration)
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
        model_path: &str,
        cache: Arc<ValueFuncCache<Game>>,
        cpu: bool,
    ) -> Box<dyn ValueFunction<Game>>;
}

struct PlayerBuilder<Game: IGame> {
    network_builder: Arc<dyn INNetworkBuilder<Game>>,
    model_path: String,
    sim_num: u32,
    explore_factor: f32,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    cache: Arc<ValueFuncCache<Game>>,
    metrics: Arc<Metrics>,
    cpu: bool,
}

impl<Game: IGame> PlayerBuilder<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        network_builder: Arc<dyn INNetworkBuilder<Game>>,
        model_path: String,
        sim_num: u32,
        explore_factor: f32,
        prior_noise_alpha: f32,
        prior_noise_epsilon: f32,
        cache_size: usize,
        metrics: Arc<Metrics>,
        cpu: bool,
    ) -> Self {
        Self {
            network_builder,
            model_path,
            sim_num,
            explore_factor,
            prior_noise_alpha,
            prior_noise_epsilon,
            cache: Arc::new(ValueFuncCache::new(cache_size)),
            metrics,
            cpu,
        }
    }
}

impl<Game: IGame> Builder<MCTSPlayer<Game>> for PlayerBuilder<Game> {
    fn build(&self) -> MCTSPlayer<Game> {
        let mut value_func: Box<dyn ValueFunction<Game>> =
            self.network_builder
                .build_net(&self.model_path, Arc::clone(&self.cache), self.cpu);

        value_func.set_run_duration_callback(Some(Box::new(NetRunDurationCallback {
            metrics: Arc::clone(&self.metrics),
        })));

        let mut player = MCTSPlayer::new_custom(
            self.sim_num,
            self.explore_factor,
            self.prior_noise_alpha,
            self.prior_noise_epsilon,
            value_func,
        );

        player.set_search_duration_callback(Some(Box::new(SearchDurationCallback {
            metrics: Arc::clone(&self.metrics),
        })));

        player
    }
}

pub fn run_main<Game: IGame + 'static>(
    network_builder: Box<dyn INNetworkBuilder<Game>>,
    serializer: Box<dyn DataSerializer<Game>>,
) -> std::io::Result<()> {
    let args = SelfPlayArgs::parse();
    let network_builder = Arc::from(network_builder);

    let cpu = match args.processing_unit.to_uppercase().as_str() {
        "CPU" => true,
        "GPU" => false,
        unknown_pu => panic!("unknown processing unit '{unknown_pu}'"),
    };
    let metrics = Arc::new(Metrics::new());

    let player1_builder = Arc::new(PlayerBuilder::new(
        Arc::clone(&network_builder),
        args.model1_path.clone(),
        args.sim_num,
        args.explore_factor,
        args.prior_noise_alpha,
        args.prior_noise_epsilon,
        args.cache_size,
        Arc::clone(&metrics),
        cpu,
    ));

    let player2_builder = if args.model1_path == args.model2_path {
        Arc::clone(&player1_builder)
    } else {
        Arc::new(PlayerBuilder::new(
            Arc::clone(&network_builder),
            args.model2_path,
            args.sim_num,
            args.explore_factor,
            args.prior_noise_alpha,
            args.prior_noise_epsilon,
            args.cache_size,
            Arc::clone(&metrics),
            cpu,
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

    if args.summary_file != "_NONE_" {
        fs::write(
            args.summary_file,
            json::object! {
                player1_wins: result.w1,
                player2_wins: result.w2,
                draws: result.d,
                net_run_duration_average_us: metrics.get_net_run_duration_average().as_micros() as u64,
                search_duration_average_ms: metrics.get_search_duration_average().as_millis() as u64,
            }
            .dump(),
        )?;
    }

    Ok(())
}
