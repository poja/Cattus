use crate::game::common::IGame;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

struct PositionCache<Game: IGame> {
    #[allow(clippy::type_complexity)]
    map: HashMap<Game::Position, (Vec<(Game::Move, f32)>, f32)>,
    deque: VecDeque<Game::Position>,
}

pub struct ValueFuncCache<Game: IGame> {
    lock: RwLock<PositionCache<Game>>,
    max_size: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl<Game: IGame> ValueFuncCache<Game> {
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0);
        Self {
            lock: RwLock::new(PositionCache {
                map: HashMap::new(),
                deque: VecDeque::new(),
            }),
            max_size,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    pub fn get_or_compute(
        &self,
        position: &Game::Position,
        mut compute: impl FnMut(&Game::Position) -> (Vec<(Game::Move, f32)>, f32),
    ) -> (Vec<(Game::Move, f32)>, f32) {
        // Acquire the read lock and check if the position is in the cache
        {
            let cache = self.lock.read().unwrap();
            if let Some(cached_val) = cache.map.get(position) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return cached_val.clone();
            }
        }

        // Compute without holding any lock
        let computed_val = compute(position);

        // Acquire the write lock, and update the cache
        {
            let mut cache = self.lock.write().unwrap();
            // Check again for the result in the cache, maybe it was added between the read and write locks acquires
            if let Some(cached_val) = cache.map.get(position) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                let cached_val = cached_val.clone();
                /* We would like to assert (computed_val == cached_val), but this is highly unreliable due to */
                /* floating points calculation errors. */
                /* This is more significant when the number of layers and params in the model is large, and it is */
                /* even more significant on the beginning of the training process, where the model contains random */
                /* values which cause very large or very small numbers. */
                return cached_val;
            }

            // Remove oldest cached elements if needed
            while cache.deque.len() >= self.max_size {
                let pos = cache.deque.pop_front().unwrap();
                cache.map.remove(&pos);
            }

            // Insert newly computed element to cache
            cache.map.insert(*position, computed_val.clone());
            cache.deque.push_back(*position);
            self.misses.fetch_add(1, Ordering::Relaxed);
            computed_val
        }
    }

    pub fn get_hits_counter(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    pub fn get_misses_counter(&self) -> usize {
        self.misses.load(Ordering::SeqCst)
    }
}
