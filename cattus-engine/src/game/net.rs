use super::mcts::NetStatistics;
use super::model::Model;
use crate::game::cache::ValueFuncCache;
use crate::game::common::{GameBitboard, GameColor, GameMove, GamePosition, IGame};
use crate::util::Device;
use itertools::Itertools;
use ndarray::{Array2, Array4};
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub struct TwoHeadedNetBase<Game: IGame> {
    model: Model,
    cache: Option<Arc<ValueFuncCache<Game>>>,

    next_batch: Mutex<Arc<Batch<Game>>>,
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
            next_batch: Mutex::new(Arc::new(Batch::new())),
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

        let (moves_scores, vals) = outputs.into_iter().collect_tuple().unwrap();
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
            let mut batch_res = self.run_net(planes_to_tensor::<Game>(&[planes]));
            let sample_res = std::mem::take(&mut batch_res[0]);
            return to_result(sample_res);
        }

        let mut attempt = 0;
        let (batch_ptr, sample_idx) = loop {
            {
                let mut next_batch = self.next_batch.lock().unwrap();
                let batch_ptr = next_batch.clone();
                let mut batch = batch_ptr.inner.lock().unwrap();

                if let BatchInner::Collect(batch_samples) = &mut *batch {
                    let sample_idx = batch_samples.len();
                    batch_samples.push(planes);
                    if batch_samples.len() < self.max_batch_size {
                        drop(batch);
                        break (batch_ptr, sample_idx);
                    }

                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Collect as u8,
                            BatchState::Compute as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only who has the batch inner lock can change the state to 'compute'
                    *next_batch = Arc::new(Batch::new());
                    drop(next_batch);

                    // drop(batch_samples);
                    let samples = std::mem::replace(&mut *batch, BatchInner::Compute);
                    let BatchInner::Collect(samples) = samples else {
                        unreachable!()
                    };
                    drop(batch);

                    let mut batch_res = self.run_net(planes_to_tensor::<Game>(&samples));
                    let sample_res = std::mem::take(&mut batch_res[sample_idx]);
                    {
                        let mut batch = batch_ptr.inner.lock().unwrap();
                        *batch = BatchInner::Done(batch_res);
                    };
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Compute as u8,
                            BatchState::Done as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only the computed thread set 'done' state
                    return to_result(sample_res);
                }
            }
            attempt += 1;
            if attempt < 3 {
                thread::yield_now();
            } else {
                thread::sleep(Duration::from_micros(100));
            }
        };

        let wait_start_time = Instant::now();
        loop {
            thread::sleep(Duration::from_micros(500));

            match batch_ptr.state.load(Ordering::SeqCst) {
                s if s == BatchState::Done as u8 => {
                    let mut batch = batch_ptr.inner.lock().unwrap();
                    let BatchInner::Done(batch_res) = &mut *batch else {
                        unreachable!()
                    };
                    let sample_res = std::mem::take(&mut batch_res[sample_idx]);
                    return to_result(sample_res);
                }
                s if s == BatchState::Collect as u8 => {
                    if wait_start_time.elapsed() < Duration::from_millis(3) {
                        continue;
                    }
                    let mut batch = batch_ptr.inner.lock().unwrap();
                    if !matches!(&*batch, BatchInner::Collect(_)) {
                        continue;
                    }
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Collect as u8,
                            BatchState::Compute as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only who has the batch inner lock can change the state to 'compute'
                    {
                        let mut next_batch = self.next_batch.lock().unwrap();
                        *next_batch = Arc::new(Batch::new());
                    }

                    let samples = std::mem::replace(&mut *batch, BatchInner::Compute);
                    let BatchInner::Collect(samples) = samples else {
                        unreachable!()
                    };
                    drop(batch);

                    let mut batch_res = self.run_net(planes_to_tensor::<Game>(&samples));
                    let sample_res = std::mem::take(&mut batch_res[sample_idx]);
                    {
                        let mut batch = batch_ptr.inner.lock().unwrap();
                        *batch = BatchInner::Done(batch_res);
                    };
                    batch_ptr
                        .state
                        .compare_exchange(
                            BatchState::Compute as u8,
                            BatchState::Done as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap(); // can not fail, only the computed thread set 'done' state
                    return to_result(sample_res);
                }
                s if s == BatchState::Compute as u8 => {}
                _ => unreachable!(),
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

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum BatchState {
    /* The batch is currently collecting samples, until it has batch_size of them */
    Collect,
    /* The batch is currently computed by one of the threads */
    Compute,
    /* The batch computation is done, the results are available to all threads */
    Done,
}

/* The size of each batch that will be send to the GPU */
/* On CPU, no batches are needed */
const GPU_BATCH_SIZE: usize = 16;

struct Batch<Game: IGame> {
    inner: Mutex<BatchInner<Game>>,
    state: AtomicU8,
}
enum BatchInner<Game: IGame> {
    Collect(Vec<Vec<Game::Bitboard>>),
    Compute,
    Done(Vec<(Vec<f32>, f32)>),
}
impl<Game: IGame> Batch<Game> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(BatchInner::Collect(vec![])),
            state: AtomicU8::new(BatchState::Collect as u8),
        }
    }
}

struct Statistics {
    activation_count: usize,
    run_duration_sum: Duration,
    batch_size_sum: usize,
}
