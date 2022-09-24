use crate::game::common::IGame;
use crate::game::mcts::{MCTSPlayer, ValueFunction};
use crate::game::self_play::{DataSerializer, PlayerBuilder, SelfPlayRunner};
use clap::Parser;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[clap(about, long_about = None)]
struct SelfPlayArgs {
    #[clap(long)]
    model1_path: String,
    #[clap(long)]
    model2_path: String,
    #[clap(long, default_value = "10")]
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
    sim_count: u32,
    #[clap(long, default_value = "1.41421")]
    explore_factor: f32,
    #[clap(long, default_value = "1")]
    threads: u32,
}

pub trait ValueFunctionBuilder<Game: IGame>: Sync + Send {
    fn new_value_func(&self, model_path: &String) -> Box<dyn ValueFunction<Game>>;
}

struct PlayerBuilderImpl<Game: IGame> {
    value_func_builder: Arc<dyn ValueFunctionBuilder<Game>>,
    model_path: String,
    sim_count: u32,
    explore_factor: f32,
}

impl<Game: IGame> PlayerBuilderImpl<Game> {
    fn new(
        value_func_builder: Arc<dyn ValueFunctionBuilder<Game>>,
        model_path: String,
        sim_count: u32,
        explore_factor: f32,
    ) -> Self {
        Self {
            value_func_builder: value_func_builder,
            model_path: model_path,
            sim_count: sim_count,
            explore_factor: explore_factor,
        }
    }
}

impl<Game: IGame> PlayerBuilder<Game> for PlayerBuilderImpl<Game> {
    fn new_player(&self) -> MCTSPlayer<Game> {
        let value_func: Box<dyn ValueFunction<Game>> =
            self.value_func_builder.new_value_func(&self.model_path);
        return MCTSPlayer::new_custom(self.sim_count, self.explore_factor, value_func);
    }
}

pub fn run_main<Game: IGame + 'static>(
    value_func_builder: Box<dyn ValueFunctionBuilder<Game>>,
    serializer: Box<dyn DataSerializer<Game>>,
) -> std::io::Result<()> {
    let args = SelfPlayArgs::parse();

    let value_func_builder = Arc::from(value_func_builder);
    let player1_builder = Box::new(PlayerBuilderImpl::new(
        Arc::clone(&value_func_builder),
        args.model1_path,
        args.sim_count,
        args.explore_factor,
    ));
    let player2_builder = Box::new(PlayerBuilderImpl::new(
        Arc::clone(&value_func_builder),
        args.model2_path,
        args.sim_count,
        args.explore_factor,
    ));

    let self_player =
        SelfPlayRunner::new(player1_builder, player2_builder, serializer, args.threads);
    return self_player.generate_data(
        args.games_num,
        &args.out_dir1,
        &args.out_dir2,
        &args.data_entries_prefix,
        &args.result_file,
    );
}
