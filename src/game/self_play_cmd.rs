use crate::game::common::IGame;
use crate::game::mcts::{MCTSPlayer, ValueFunction};
use crate::game::self_play::{DataSerializer, SelfPlayRunner};
use crate::utils::Builder;
use clap::Parser;
use std::sync::Arc;

use super::cache::ValueFuncCache;

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
    #[clap(long, default_value = "")]
    data_entries_prefix: String,
    #[clap(long, default_value = "_NONE_")]
    result_file: String,
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
            cpu,
        }
    }
}

impl<Game: IGame> Builder<MCTSPlayer<Game>> for PlayerBuilder<Game> {
    fn build(&self) -> MCTSPlayer<Game> {
        let value_func: Box<dyn ValueFunction<Game>> =
            self.network_builder
                .build_net(&self.model_path, Arc::clone(&self.cache), self.cpu);
        MCTSPlayer::new_custom(
            self.sim_num,
            self.explore_factor,
            self.prior_noise_alpha,
            self.prior_noise_epsilon,
            value_func,
        )
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

    let player1_builder = Arc::new(PlayerBuilder::new(
        Arc::clone(&network_builder),
        args.model1_path.clone(),
        args.sim_num,
        args.explore_factor,
        args.prior_noise_alpha,
        args.prior_noise_epsilon,
        args.cache_size,
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
            cpu,
        ))
    };

    SelfPlayRunner::new(
        player1_builder,
        player2_builder,
        args.temperature_policy,
        Arc::from(serializer),
        args.threads,
    )
    .generate_data(
        args.games_num,
        &args.out_dir1,
        &args.out_dir2,
        &args.data_entries_prefix,
        &args.result_file,
    )
}
