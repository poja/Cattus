use itertools::Itertools;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::game::cache::ValueFuncCache;
use crate::game::common::{GameBitboard, GameColor, GameMove, GamePosition, IGame};

use super::mcts::NetStatistics;

struct BatchData<Game: IGame> {
    samples: Vec<Vec<Game::Bitboard>>,
    res: Option<Vec<(f32, Vec<f32>)>>,
}

/* The batch is currently collecting samples, until it has batch_size of them */
const BATCH_STATE_COLLECT: u8 = 0;
/* The batch is currently computed by one of the threads */
const BATCH_STATE_COMPUTATION: u8 = 1;
/* The batch computation is done, the results are available to all threads */
const BATCH_STATE_DONE: u8 = 2;

/* The size of each batch that will be send to the GPU */
/* On CPU, no batches are needed */
const GPU_BATCH_SIZE: usize = 16;
const BATCH_WAIT_UNTIL_FULL_POLL_FREQ_US: u64 = 300;
const BATCH_WAIT_UNTIL_FULL_TIMEOUT_US: u64 = 3000;

struct Batch<Game: IGame> {
    data: Mutex<BatchData<Game>>,
    state: AtomicU8,
}
impl<Game: IGame> Batch<Game> {
    fn new_empty() -> Self {
        Self {
            data: Mutex::new(BatchData {
                samples: vec![],
                res: None,
            }),
            state: AtomicU8::new(BATCH_STATE_COLLECT),
        }
    }
}

struct NextBatchManager<Game: IGame> {
    next_batch: Arc<Batch<Game>>,
    batch_idx: usize,
}
impl<Game: IGame> NextBatchManager<Game> {
    fn new_empty() -> Self {
        Self {
            next_batch: Arc::new(Batch::new_empty()),
            batch_idx: 0,
        }
    }

    /// Release the current 'next_batch'.
    /// The thread calling this function has the responsibility to compute the result of the batch.
    fn advance_batch(&mut self) {
        self.next_batch = Arc::new(Batch::new_empty());
        self.batch_idx += 1;
    }
}

struct Statistics {
    activation_count: usize,
    run_duration_sum: Duration,
    batch_size_sum: usize,
}

pub struct TwoHeadedNetBase<Game: IGame, const CPU: bool> {
    bundle: SavedModelBundle,
    input_op: Operation,
    value_head: Operation,
    policy_head: Operation,
    cache: Option<Arc<ValueFuncCache<Game>>>,

    next_batch_manager: Mutex<NextBatchManager<Game>>,
    max_batch_size: usize,

    stats: Mutex<Statistics>,
}

impl<Game: IGame, const CPU: bool> TwoHeadedNetBase<Game, CPU> {
    pub fn new(model_path: &str, cache: Option<Arc<ValueFuncCache<Game>>>) -> Self {
        // Load saved model bundle (session state + meta_graph data)
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path)
                .expect("Can't load saved model");

        // Get signature metadata from the model bundle
        let signature = bundle
            .meta_graph_def()
            .get_signature("serving_default")
            .unwrap();

        // Get input/output info
        let input_info = signature.get_input("input_planes").unwrap();
        let output_scalar_info = signature.get_output("value_head").unwrap();
        let output_probs_info = signature.get_output("policy_head").unwrap();

        // Get input/output ops from graph
        let input_op = graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        let value_head = graph
            .operation_by_name_required(&output_scalar_info.name().name)
            .unwrap();
        let policy_head = graph
            .operation_by_name_required(&output_probs_info.name().name)
            .unwrap();

        Self {
            bundle,
            input_op,
            value_head,
            policy_head,
            cache,
            next_batch_manager: Mutex::new(NextBatchManager::new_empty()),
            max_batch_size: if CPU { 1 } else { GPU_BATCH_SIZE },
            stats: Mutex::new(Statistics {
                activation_count: 0,
                run_duration_sum: Duration::ZERO,
                batch_size_sum: 0,
            }),
        }
    }

    pub fn run_net(&self, input: Tensor<f32>) -> Vec<(f32, Vec<f32>)> {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output_scalar = args.request_fetch(&self.value_head, 1);
        let output_moves_scores = args.request_fetch(&self.policy_head, 0);

        let net_run_begin = Instant::now();
        self.bundle
            .session
            .run(&mut args)
            .expect("Error occurred during calculations");
        let run_duration = net_run_begin.elapsed();

        let vals: Tensor<f32> = args.fetch(output_scalar).unwrap();
        let moves_scores: Tensor<f32> = args.fetch(output_moves_scores).unwrap();
        let batch_size = vals.shape()[0].unwrap() as usize;

        let ret = (0..batch_size)
            .into_iter()
            .map(|sample_idx| {
                let mut val: f32 = vals[sample_idx];
                if !val.is_finite() {
                    val = 0.0;
                }

                let sample_len = moves_scores.len() / batch_size;
                let sample_scores = moves_scores
                    [sample_idx * sample_len..((sample_idx + 1) * sample_len)]
                    .iter()
                    .map(|s| if s.is_finite() { *s } else { f32::MIN })
                    .collect_vec();

                (val, sample_scores)
            })
            .collect_vec();

        // update stats
        let mut stats = self.stats.lock().unwrap();
        stats.activation_count += 1;
        stats.batch_size_sum += batch_size;
        stats.run_duration_sum += run_duration;

        ret
    }

    pub fn evaluate(
        &self,
        position: &Game::Position,
        to_planes: impl Fn(&Game::Position) -> Vec<Game::Bitboard>,
    ) -> (f32, Vec<(Game::Move, f32)>) {
        let (position, is_flipped) = flip_pos_if_needed(*position);

        let res = if let Some(cache) = &self.cache {
            cache.get_or_compute(&position, |pos| self.evaluate_impl(pos, &to_planes))
        } else {
            self.evaluate_impl(&position, &to_planes)
        };

        flip_score_if_needed(res, is_flipped)
    }

    fn evaluate_impl(
        &self,
        pos: &Game::Position,
        to_planes: &impl Fn(&Game::Position) -> Vec<Game::Bitboard>,
    ) -> (f32, Vec<(Game::Move, f32)>) {
        /* Add a new sample to the current batch */
        let batch;
        let batch_idx;
        let sample_idx;
        let mut run_net = {
            let mut next_batch_manager = self.next_batch_manager.lock().unwrap();
            batch = Arc::clone(&(next_batch_manager.next_batch));
            batch_idx = next_batch_manager.batch_idx;
            let batch_is_full = {
                let mut batch_data = batch.data.lock().unwrap();
                assert!(batch_data.samples.len() < self.max_batch_size);
                sample_idx = batch_data.samples.len();
                batch_data.samples.push(to_planes(pos));
                batch_data.samples.len() >= self.max_batch_size
            };
            if batch_is_full {
                let cmp_res = batch.state.compare_exchange(
                    BATCH_STATE_COLLECT,
                    BATCH_STATE_COMPUTATION,
                    Ordering::SeqCst,
                    /* doesn't matter */ Ordering::Relaxed,
                );
                assert!(cmp_res.is_ok());

                next_batch_manager.advance_batch();

                debug_assert!({
                    let next_batch_data = next_batch_manager.next_batch.data.lock().unwrap();
                    next_batch_data.samples.len() == 0
                });
                debug_assert!({
                    let batch_data = batch.data.lock().unwrap();
                    batch_data.samples.len() == self.max_batch_size
                });
            }
            batch_is_full
        };

        /*
         * If the batch was not full, we wait until another thread will fill it and run the network on
         * the whole batch. If too much time pass, we compute it ourselves.
         */
        let sample_res;
        let wait_start_time = Instant::now();
        let mut maybe_compute_ourselves = true;
        loop {
            if run_net {
                /* The batch is either full or we waited too much. Run the network on batch */
                let mut batch_data = batch.data.lock().unwrap();
                assert!(batch_data.res.is_none());
                let res = self.run_net(planes_to_tensor::<Game, CPU>(&batch_data.samples));
                sample_res = res[sample_idx].clone();
                batch_data.res = Some(res);

                let cmp_res = batch.state.compare_exchange(
                    BATCH_STATE_COMPUTATION,
                    BATCH_STATE_DONE,
                    Ordering::SeqCst,
                    /* doesn't matter */ Ordering::Relaxed,
                );
                assert!(cmp_res.is_ok());
                break;
            }

            /* wait a little bit */
            std::thread::sleep(Duration::from_micros(BATCH_WAIT_UNTIL_FULL_POLL_FREQ_US));
            {
                /* check if another thread computed the result */
                if batch.state.load(Ordering::SeqCst) == BATCH_STATE_DONE {
                    let batch_data = batch.data.lock().unwrap();
                    let samples_result = batch_data.res.as_ref().unwrap();
                    sample_res = samples_result[sample_idx].clone();
                    break;
                }
            }

            if maybe_compute_ourselves
                && wait_start_time.elapsed()
                    >= Duration::from_micros(BATCH_WAIT_UNTIL_FULL_TIMEOUT_US)
            {
                /* Too much time passed. We compute the result ourselves */
                let mut next_batch_manager = self.next_batch_manager.lock().unwrap();
                if batch_idx == next_batch_manager.batch_idx {
                    /* Our batch is the 'next_batch', meaning no other thread started computing the result */
                    /* create a new empty batch as next_batch and prepare to run the network ourselves */
                    let cmp_res = batch.state.compare_exchange(
                        BATCH_STATE_COLLECT,
                        BATCH_STATE_COMPUTATION,
                        Ordering::SeqCst,
                        /* doesn't matter */ Ordering::Relaxed,
                    );
                    assert!(cmp_res.is_ok());
                    next_batch_manager.advance_batch();
                    run_net = true;
                } else {
                    /* Our batch is not the 'next_batch', meaning other thread is going to compute the result */
                    maybe_compute_ourselves = false;
                }
            }
        }

        let (val, move_scores) = sample_res;
        let moves = pos.get_legal_moves();
        let moves_probs = calc_moves_probs::<Game>(moves, &move_scores);
        (val, moves_probs)
    }

    pub fn get_statistics(&self) -> NetStatistics {
        let stats = self.stats.lock().unwrap();
        NetStatistics {
            activation_count: Some(stats.activation_count),
            run_duration_average: Some(stats.run_duration_sum / stats.activation_count as u32),
            batch_size_average: Some(stats.batch_size_sum as f32 / stats.activation_count as f32),
        }
    }
}

pub fn calc_moves_probs<Game: IGame>(
    moves: Vec<Game::Move>,
    move_scores: &[f32],
) -> Vec<(Game::Move, f32)> {
    let moves_scores = moves
        .iter()
        .map(|m| move_scores[m.to_nn_idx()])
        .collect_vec();

    // Softmax normalization
    let max_p = moves_scores.iter().cloned().fold(f32::MIN, f32::max);
    let scores = moves_scores
        .into_iter()
        .map(|p| (p - max_p).exp())
        .collect_vec();
    let p_sum: f32 = scores.iter().sum();
    let probs = scores.into_iter().map(|p| p / p_sum).collect_vec();

    moves.into_iter().zip(probs.into_iter()).collect_vec()
}

pub fn planes_to_tensor<Game: IGame, const CPU: bool>(
    samples: &Vec<Vec<Game::Bitboard>>,
) -> Tensor<f32> {
    let batch_size = samples.len() as u64;
    let planes_num = samples[0].len() as u64;
    let dims = if CPU {
        [
            batch_size,
            Game::BOARD_SIZE as u64,
            Game::BOARD_SIZE as u64,
            planes_num,
        ]
    } else {
        [
            batch_size,
            planes_num,
            Game::BOARD_SIZE as u64,
            Game::BOARD_SIZE as u64,
        ]
    };
    let mut tensor = Tensor::new(&dims);

    for (sample_idx, sample) in samples.into_iter().enumerate() {
        for (plane_idx, plane) in sample.into_iter().enumerate() {
            for r in 0..(Game::BOARD_SIZE as u64) {
                for c in 0..(Game::BOARD_SIZE as u64) {
                    let indices = if CPU {
                        [sample_idx as u64, r, c, plane_idx as u64]
                    } else {
                        [sample_idx as u64, plane_idx as u64, r, c]
                    };
                    let val = match plane.get((r as usize) * Game::BOARD_SIZE + c as usize) {
                        true => 1.0,
                        false => 0.0,
                    };
                    tensor.set(&indices, val);
                }
            }
        }
    }

    tensor
}

pub fn flip_pos_if_needed<Position: GamePosition>(pos: Position) -> (Position, bool) {
    if pos.get_turn() == GameColor::Player1 {
        (pos, false)
    } else {
        (pos.get_flip(), true)
    }
}

pub fn flip_score_if_needed<Move: GameMove>(
    net_res: (f32, Vec<(Move, f32)>),
    pos_flipped: bool,
) -> (f32, Vec<(Move, f32)>) {
    if !pos_flipped {
        net_res
    } else {
        let (val, moves_probs) = net_res;

        /* Flip scalar value */
        let val = -val;

        /* Flip moves */
        let moves_probs = moves_probs
            .into_iter()
            .map(|(m, p)| (m.get_flip(), p))
            .collect_vec();

        (val, moves_probs)
    }
}
