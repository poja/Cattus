use crate::game::common::IGame;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

struct PositionCache<Game: IGame> {
    #[allow(clippy::type_complexity)]
    map: HashMap<Game::Position, (Vec<(Game::Move, f32)>, f32)>,
    deque: VecDeque<Game::Position>,
}

pub struct ValueFuncCache<Game: IGame> {
    lock: RwLock<PositionCache<Game>>,
    max_size: usize,
    hits: metrics::Counter,
    misses: metrics::Counter,
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
            hits: metrics::counter!("cache.hits"),
            misses: metrics::counter!("cache.misses"),
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
                self.hits.increment(1);
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
                self.hits.increment(1);
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
            self.misses.increment(1);
            computed_val
        }
    }
}
