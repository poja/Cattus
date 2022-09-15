use itertools::Itertools;

use crate::game::common::IGame;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::{fs, thread};

use super::common::GameColor;
use super::mcts::MCTSPlayer;

pub trait PlayerBuilder<Game: IGame>: Sync + Send {
    fn new_player(&self) -> MCTSPlayer<Game>;
}

pub struct PlayerComparator<Game: IGame> {
    player1_builder: Arc<dyn PlayerBuilder<Game>>,
    player2_builder: Arc<dyn PlayerBuilder<Game>>,
    thread_num: u32,
}

impl<Game: IGame + 'static> PlayerComparator<Game> {
    pub fn new(
        player1_builder: Box<dyn PlayerBuilder<Game>>,
        player2_builder: Box<dyn PlayerBuilder<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_builder: Arc::from(player1_builder),
            player2_builder: Arc::from(player2_builder),
            thread_num: thread_num,
        }
    }

    pub fn compare_players(&self, games_num: u32, result_file: &String) -> std::io::Result<()> {
        let player1_wins = Arc::new(AtomicU32::new(0));
        let player2_wins = Arc::new(AtomicU32::new(0));
        let draws = Arc::new(AtomicU32::new(0));

        let job_builder = |thread_idx| {
            let thread_game_num = games_num * (thread_idx + 1) / self.thread_num
                - games_num * thread_idx / self.thread_num;

            let worker = ComparatorWorker::new(
                Arc::clone(&self.player1_builder),
                Arc::clone(&self.player2_builder),
                thread_game_num,
                Arc::clone(&player1_wins),
                Arc::clone(&player2_wins),
                Arc::clone(&draws),
            );

            return move || worker.compare_players();
        };

        /* Spawn thread_num-1 to jobs [1..thread_num-1] */
        let threads = (1..self.thread_num)
            .map(|thread_idx| thread::spawn(job_builder(thread_idx)))
            .collect_vec();

        /* Use current thread to do job 0 */
        job_builder(0)();

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        let json_obj = json::object! {
            player1_wins: player1_wins.load(Ordering::Relaxed),
            player2_wins: player2_wins.load(Ordering::Relaxed),
            draws: draws.load(Ordering::Relaxed),
        };

        let json_str = json_obj.dump();
        fs::write(result_file, json_str)?;

        return Ok(());
    }
}

struct ComparatorWorker<Game: IGame> {
    player1_builder: Arc<dyn PlayerBuilder<Game>>,
    player2_builder: Arc<dyn PlayerBuilder<Game>>,
    games_num: u32,
    player1_wins: Arc<AtomicU32>,
    player2_wins: Arc<AtomicU32>,
    draws: Arc<AtomicU32>,
}

impl<Game: IGame> ComparatorWorker<Game> {
    fn new(
        player1_builder: Arc<dyn PlayerBuilder<Game>>,
        player2_builder: Arc<dyn PlayerBuilder<Game>>,
        games_num: u32,
        player1_wins: Arc<AtomicU32>,
        player2_wins: Arc<AtomicU32>,
        draws: Arc<AtomicU32>,
    ) -> Self {
        Self {
            player1_builder: player1_builder,
            player2_builder: player2_builder,
            games_num: games_num,
            player1_wins: player1_wins,
            player2_wins: player2_wins,
            draws: draws,
        }
    }

    fn compare_players(&self) {
        let mut player1 = self.player1_builder.new_player();
        let mut player2 = self.player2_builder.new_player();
        for _ in 0..self.games_num {
            let (_final_pos, score) = Game::new().play_until_over(&mut player1, &mut player2);
            match score {
                Some(GameColor::Player1) => &self.player1_wins,
                Some(GameColor::Player2) => &self.player2_wins,
                None => &self.draws,
            }
            .fetch_add(1, Ordering::Relaxed);
        }
    }
}
