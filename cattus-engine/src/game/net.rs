use super::mcts::NetStatistics;
use super::model::Model;
use crate::game::cache::ValueFuncCache;
use crate::game::common::{GameBitboard, GameColor, GameMove, GamePosition, IGame};
use crate::utils::Device;
use itertools::Itertools;
use ndarray::{Array2, Array4};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum BatchState {
    /* The batch is currently collecting samples, until it has batch_size of them */
    Collect,
    /* The batch is currently computed by one of the threads */
    Computation,
    /* The batch computation is done, the results are available to all threads */
    Done,
}

/* The size of each batch that will be send to the GPU */
/* On CPU, no batches are needed */
const GPU_BATCH_SIZE: usize = 16;
const BATCH_WAIT_UNTIL_FULL_POLL_FREQ_US: u64 = 500;
const BATCH_WAIT_UNTIL_FULL_TIMEOUT_US: u64 = 3000;

struct Batch<Game: IGame> {
    samples: Vec<Vec<Game::Bitboard>>,
    res: Option<Vec<(Vec<f32>, f32)>>,
    state: BatchState,
}
impl<Game: IGame> Batch<Game> {
    fn new() -> Self {
        Self {
            samples: vec![],
            res: None,
            state: BatchState::Collect,
        }
    }
}

struct NextBatchManager<Game: IGame> {
    next_batch: Arc<Mutex<Batch<Game>>>,
}
impl<Game: IGame> NextBatchManager<Game> {
    fn new_empty() -> Self {
        Self {
            next_batch: Arc::new(Mutex::new(Batch::new())),
        }
    }

    /// Release the current 'next_batch'.
    /// The thread calling this function has the responsibility to compute the result of the batch.
    fn advance_batch(&mut self) {
        self.next_batch = Arc::new(Mutex::new(Batch::new()));
    }
}

struct Statistics {
    activation_count: usize,
    run_duration_sum: Duration,
    batch_size_sum: usize,
}

pub struct TwoHeadedNetBase<Game: IGame> {
    model: Model,
    cache: Option<Arc<ValueFuncCache<Game>>>,

    next_batch_manager: Mutex<NextBatchManager<Game>>,
    max_batch_size: usize,

    stats: Mutex<Statistics>,
}

impl<Game: IGame> TwoHeadedNetBase<Game> {
    pub fn new(
        model_path: impl AsRef<Path>,
        device: Device,
        cache: Option<Arc<ValueFuncCache<Game>>>,
    ) -> Self {
        Self {
            model: Model::new(model_path),
            cache,
            next_batch_manager: Mutex::new(NextBatchManager::new_empty()),
            max_batch_size: if matches!(device, Device::Cpu) {
                1
            } else {
                GPU_BATCH_SIZE
            },
            stats: Mutex::new(Statistics {
                activation_count: 0,
                run_duration_sum: Duration::ZERO,
                batch_size_sum: 0,
            }),
        }
    }

    pub fn run_net(&self, input: Array4<f32>) -> Vec<(Vec<f32>, f32)> {
        let net_run_begin = Instant::now();
        let outputs = self.model.run([input.into_dyn()].to_vec());
        let run_duration = net_run_begin.elapsed();

        let outputs: [_; 2] = outputs.try_into().unwrap();
        let (moves_scores, vals) = outputs.into();
        let moves_scores: Array2<f32> = moves_scores.into_dimensionality().unwrap();
        let vals: Array2<f32> = vals.into_dimensionality().unwrap();

        let batch_size = vals.len();
        let ret = moves_scores
            .rows()
            .into_iter()
            .zip(vals)
            .map(|(sample_scores, val)| {
                let mut sample_scores = sample_scores.to_vec();
                for s in sample_scores.iter_mut() {
                    if !s.is_finite() {
                        *s = f32::MIN;
                    }
                }
                (sample_scores, val)
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
    ) -> (Vec<(Game::Move, f32)>, f32) {
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
    ) -> (Vec<(Game::Move, f32)>, f32) {
        let planes = to_planes(pos);

        let to_result = |sample_res: (Vec<f32>, f32)| {
            let (move_scores, val) = sample_res;
            let moves = pos.get_legal_moves();
            let moves_probs = calc_moves_probs::<Game>(moves, &move_scores);
            (moves_probs, val)
        };

        if self.max_batch_size <= 1 {
            let mut brach_res = self.run_net(planes_to_tensor::<Game>(&[planes]));
            let sample_res = std::mem::take(&mut brach_res[0]);
            return to_result(sample_res);
        }

        let (batch_ptr, sample_idx) = loop {
            let mut next_batch_manager = self.next_batch_manager.lock().unwrap();
            let batch_ptr = next_batch_manager.next_batch.clone();
            let mut batch = batch_ptr.lock().unwrap();
            if batch.state == BatchState::Collect {
                let sample_idx = batch.samples.len();
                batch.samples.push(planes);
                if batch.samples.len() < self.max_batch_size {
                    drop(batch);
                    break (batch_ptr, sample_idx);
                }

                batch.state = BatchState::Computation;
                next_batch_manager.advance_batch();
                drop(next_batch_manager);

                let samples = std::mem::take(&mut batch.samples);
                drop(batch);

                let mut brach_res = self.run_net(planes_to_tensor::<Game>(&samples));
                let sample_res = std::mem::take(&mut brach_res[sample_idx]);
                {
                    let mut batch = batch_ptr.lock().unwrap();
                    batch.res = Some(brach_res);
                    batch.state = BatchState::Done;
                };
                return to_result(sample_res);
            }
            thread::sleep(Duration::from_micros(BATCH_WAIT_UNTIL_FULL_POLL_FREQ_US));
        };

        let wait_start_time = Instant::now();
        loop {
            thread::sleep(Duration::from_micros(BATCH_WAIT_UNTIL_FULL_POLL_FREQ_US));
            let mut batch = batch_ptr.lock().unwrap();
            if batch.state == BatchState::Done {
                let brach_res = batch.res.as_mut().unwrap();
                let sample_res = std::mem::take(&mut brach_res[sample_idx]);
                return to_result(sample_res);
            }

            if batch.state == BatchState::Collect
                && wait_start_time.elapsed()
                    >= Duration::from_micros(BATCH_WAIT_UNTIL_FULL_TIMEOUT_US)
            {
                self.next_batch_manager.lock().unwrap().advance_batch();

                batch.state = BatchState::Computation;

                let samples = std::mem::take(&mut batch.samples);
                drop(batch);

                let mut brach_res = self.run_net(planes_to_tensor::<Game>(&samples));
                let sample_res = std::mem::take(&mut brach_res[sample_idx]);
                {
                    let mut batch = batch_ptr.lock().unwrap();
                    batch.res = Some(brach_res);
                    batch.state = BatchState::Done;
                };
                return to_result(sample_res);
            }
        }
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

    moves.into_iter().zip(probs).collect_vec()
}

pub fn planes_to_tensor<Game: IGame>(samples: &[Vec<Game::Bitboard>]) -> Array4<f32> {
    let batch_size = samples.len();
    let planes_num = samples[0].len();
    let dims = (batch_size, planes_num, Game::BOARD_SIZE, Game::BOARD_SIZE);
    let mut tensor: Array4<f32> = Array4::zeros(dims);

    for (b, sample) in samples.iter().enumerate() {
        for (c, plane) in sample.iter().enumerate() {
            for h in 0..(Game::BOARD_SIZE) {
                for w in 0..(Game::BOARD_SIZE) {
                    tensor[(b, c, h, w)] = match plane.get(h * Game::BOARD_SIZE + w) {
                        true => 1.0,
                        false => 0.0,
                    };
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
    net_res: (Vec<(Move, f32)>, f32),
    pos_flipped: bool,
) -> (Vec<(Move, f32)>, f32) {
    if !pos_flipped {
        return net_res;
    }
    let (moves_probs, val) = net_res;

    /* Flip scalar value */
    let val = -val;

    /* Flip moves */
    let moves_probs = moves_probs
        .into_iter()
        .map(|(m, p)| (m.get_flip(), p))
        .collect_vec();

    (moves_probs, val)
}
