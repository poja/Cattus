use cattus::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use cattus::game::common::{GameBitboard, IGame};
use cattus::hex::hex_game::{HexBitboard, HexGame, HexMove, HexPosition};
use cattus::ttt::ttt_game::{TttBitboard, TttGame, TttMove, TttPosition};
use itertools::Itertools;
use std::fs;
use std::ops::Deref;
use std::path::{self, Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;

use cattus::game::common::{GameColor, GameMove, GamePosition};
use cattus::game::mcts::{MctsParams, MctsPlayer};
use cattus::game::net;

use crate::serialize::DataSerializer;

pub struct DataEntry<Game: IGame> {
    pub pos: Game::Position,
    pub probs: Vec<(Game::Move, f32)>,
    pub winner: Option<GameColor>,
}

impl<Game: IGame> Clone for DataEntry<Game> {
    fn clone(&self) -> Self {
        DataEntry {
            pos: self.pos,
            probs: self.probs.clone(),
            winner: self.winner,
        }
    }
}

pub struct SerializerBase;
impl SerializerBase {
    pub fn write_entry<Game: IGame>(
        planes: Vec<u64>,
        probs: Vec<(Game::Move, f32)>,
        winner: i8,
        filename: &Path,
    ) -> std::io::Result<()> {
        /* Use -1 for illegal moves */
        let mut probs_vec = vec![-1.0f32; Game::MOVES_NUM];

        /* Fill legal moves probabilities */
        for (m, prob) in probs {
            probs_vec[m.to_nn_idx()] = prob;
        }

        let u64bytes = u64::BITS as usize / 8;
        let f32bytes = /* f32::BITS */ 32 / 8;
        let i8bytes = i8::BITS as usize / 8;
        let size = planes.len() * u64bytes + probs_vec.len() * f32bytes + i8bytes;
        let mut bytes = Vec::with_capacity(size);

        /* Serialized in little indian format, should deserialized the same */
        bytes.extend(planes.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(probs_vec.into_iter().flat_map(|p| p.to_le_bytes()));
        bytes.extend(winner.to_le_bytes());
        assert!(bytes.len() == size);

        /* Write to file */
        fs::write(filename, bytes)
    }
}

#[derive(Copy, Clone)]
pub struct GamesResults {
    pub w1: u32,
    pub w2: u32,
    pub d: u32,
}

pub struct SelfPlayRunner<Game: IGame> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    temperature_policy: String,
    serializer: Arc<dyn DataSerializer<Game>>,
    thread_num: usize,
}

impl<Game: SelfPlayGame + 'static> SelfPlayRunner<Game> {
    pub fn new(
        player1_params: MctsParams<Game>,
        player2_params: MctsParams<Game>,
        temperature_policy: String,
        serializer: Arc<dyn DataSerializer<Game>>,
        thread_num: u32,
    ) -> Self {
        assert!(thread_num > 0);
        Self {
            player1_params,
            player2_params,
            temperature_policy,
            serializer,
            thread_num: thread_num as usize,
        }
    }

    pub fn generate_data(
        &self,
        games_num: usize,
        output_dir1: &Path,
        output_dir2: &Path,
    ) -> std::io::Result<GamesResults> {
        assert!(games_num % 2 == 0, "Games num should be a multiple of 2");

        /* Create output dir if doesn't exists */
        for output_dir in [output_dir1, output_dir2] {
            if !path::Path::new(output_dir).is_dir() {
                fs::create_dir_all(output_dir)?;
            }
        }

        let games_counter = Arc::new(AtomicUsize::new(0));
        let result = Arc::new(Mutex::new(GamesResults { w1: 0, w2: 0, d: 0 }));

        let job_builder = || {
            let worker = SelfPlayWorker::new(
                self.player1_params.clone(),
                self.player2_params.clone(),
                &self.temperature_policy,
                self.serializer.clone(),
                output_dir1.to_path_buf(),
                output_dir2.to_path_buf(),
                result.clone(),
                games_counter.clone(),
                games_num,
            );

            move || worker.generate_data().unwrap()
        };

        /* Spawn thread_num-1 to jobs [1..thread_num-1] */
        // TODO: add termination mechanism, detect if one of the threads panic
        let threads = (1..self.thread_num)
            .map(|_| thread::spawn(job_builder()))
            .collect_vec();

        /* Use current thread to do job 0 */
        job_builder()();

        /* Join all threads */
        for t in threads {
            t.join().unwrap();
        }

        let res = *result.lock().unwrap().deref();
        Ok(res)
    }
}

struct SelfPlayWorker<Game: IGame> {
    player1_params: MctsParams<Game>,
    player2_params: MctsParams<Game>,
    temperature_scheduler: TemperatureScheduler,
    serializer: Arc<dyn DataSerializer<Game>>,
    output_dir1: PathBuf,
    output_dir2: PathBuf,
    results: Arc<Mutex<GamesResults>>,
    games_queue: Arc<AtomicUsize>,
    games_num: usize,
}

impl<Game: SelfPlayGame> SelfPlayWorker<Game> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        player1_params: MctsParams<Game>,
        player2_params: MctsParams<Game>,
        temperature_policy: &str,
        serializer: Arc<dyn DataSerializer<Game>>,
        output_dir1: PathBuf,
        output_dir2: PathBuf,
        results: Arc<Mutex<GamesResults>>,
        games_queue: Arc<AtomicUsize>,
        games_num: usize,
    ) -> Self {
        Self {
            player1_params,
            player2_params,
            temperature_scheduler: TemperatureScheduler::from_str(temperature_policy),
            serializer,
            output_dir1,
            output_dir2,
            results,
            games_queue,
            games_num,
        }
    }

    fn generate_data(&self) -> std::io::Result<()> {
        let mut player1 = MctsPlayer::new(self.player1_params.clone());
        let mut player2 = MctsPlayer::new(self.player2_params.clone());

        loop {
            let game_idx = self
                .games_queue
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if game_idx >= self.games_num {
                break;
            }

            let mut game = Game::new();
            let mut pos_probs_pairs = Vec::new();
            let players_switch = game_idx % 2 == 1;

            let mut half_move_num = 0;
            while !game.is_over() {
                let mut player = game.get_position().get_turn();
                if players_switch {
                    player = player.opposite()
                }
                let player = match player {
                    GameColor::Player1 => &mut player1,
                    GameColor::Player2 => &mut player2,
                };

                /* Generate probabilities from MCTS player */
                player.set_temperature(
                    self.temperature_scheduler
                        .get_temperature((half_move_num / 2) as u32),
                );
                let moves = player.calc_moves_probabilities(game.get_position());
                let next_move = player.choose_move_from_probabilities(&moves).unwrap();

                /* Store probabilities */
                pos_probs_pairs.push((*game.get_position(), moves));

                /* Advance game position */
                game.play_single_turn(next_move);

                half_move_num += 1;
            }

            /* Save all data entries */
            let winner = game.get_winner();
            for (pos_idx, (pos, probs)) in pos_probs_pairs.into_iter().enumerate() {
                self.write_data_entry(game_idx, pos_idx, pos, probs, winner)?;
            }

            /* Update winning counters */
            {
                let mut results = self.results.lock().unwrap();
                let counter = match winner {
                    None => &mut results.d,
                    Some(mut player) => {
                        if players_switch {
                            player = player.opposite();
                        }
                        match player {
                            GameColor::Player1 => &mut results.w1,
                            GameColor::Player2 => &mut results.w2,
                        }
                    }
                };
                *counter += 1;
            }

            log::debug!("Game {} done", game_idx);
        }
        Ok(())
    }

    fn write_data_entry(
        &self,
        game_idx: usize,
        pos_idx: usize,
        pos: Game::Position,
        probs: Vec<(Game::Move, f32)>,
        winner: Option<GameColor>,
    ) -> std::io::Result<()> {
        let output_dir = match pos.get_turn() {
            GameColor::Player1 => [&self.output_dir1, &self.output_dir2],
            GameColor::Player2 => [&self.output_dir2, &self.output_dir1],
        }[game_idx % 2];

        let winner = GameColor::to_idx(winner) as f32;
        let (pos, is_flipped) = net::flip_pos_if_needed(pos);
        let (probs, winner) = net::flip_score_if_needed((probs, winner), is_flipped);
        let winner = GameColor::from_idx(winner as i32);

        let entries = Game::produce_transformed_data_entries(DataEntry { pos, probs, winner });
        for (transform_idx, entry) in entries.into_iter().enumerate() {
            self.serializer.serialize_data_entry(
                entry,
                &output_dir.join(format!(
                    "{game_idx:#08}_{pos_idx:#03}_{transform_idx:#02}.traindata",
                )),
            )?;
        }
        Ok(())
    }
}

struct TemperatureScheduler {
    temperatures: Vec<(u32, f32)>,
    last_temperature: f32,
}

impl TemperatureScheduler {
    /// Create a scheduler from a string describing the temperature policy
    ///
    /// # Arguments
    ///
    /// * `s` - A string representing the temperature policy. The string should contain an odd number of numbers,
    ///   with a ',' between them.
    ///
    /// The string will be split into pairs of two numbers, and a final number.
    /// Each pair should be of the form (moves_num, temperature) and the final number is the final temperature.
    /// Each pair represent an interval of moves number in which the corresponding temperature will be assigned.
    /// The pairs should be ordered by the moves_num.
    ///
    /// # Examples
    ///
    /// "1.0" means a constant temperature of 1
    /// "30,1.0,0.0" means a temperature of 1.0 for the first 30 moves, and temperature of zero after than
    /// "15,2.0,30,0.5,0.1" means a temperature of 2.0 for the first 15 moves, 0.5 in the moves 16 up to 30, and 0.1
    /// after that
    fn from_str(s: &str) -> Self {
        let s = s.split(',').collect_vec();
        assert!(s.len() % 2 == 1);

        let mut temperatures = Vec::new();
        for i in 0..((s.len() - 1) / 2) {
            let threshold = s[i * 2].parse::<u32>().unwrap();
            let temperature = s[i * 2 + 1].parse::<f32>().unwrap();
            if !temperatures.is_empty() {
                let (last_threshold, _last_temp) = temperatures.last().unwrap();
                assert!(*last_threshold < threshold);
            }
            temperatures.push((threshold, temperature));
        }
        let last_temp = s.last().unwrap().parse::<f32>().unwrap();
        Self {
            temperatures,
            last_temperature: last_temp,
        }
    }

    fn get_temperature(&self, move_num: u32) -> f32 {
        for (threshold, temperature) in &self.temperatures {
            if move_num < *threshold {
                return *temperature;
            }
        }
        self.last_temperature
    }
}

pub trait SelfPlayGame: IGame {
    fn produce_transformed_data_entries(entry: DataEntry<Self>) -> Vec<DataEntry<Self>>;
}
impl SelfPlayGame for TttGame {
    fn produce_transformed_data_entries(entry: DataEntry<Self>) -> Vec<DataEntry<Self>> {
        let transform = |e: &DataEntry<Self>, transform_sq: &dyn Fn(usize) -> usize| {
            let (board_x, board_o) = [e.pos.board_x, e.pos.board_o]
                .iter()
                .map(|b| {
                    let mut bt = TttBitboard::new();
                    for idx in 0..TttGame::BOARD_SIZE * TttGame::BOARD_SIZE {
                        bt.set(transform_sq(idx), b.get(idx));
                    }
                    bt
                })
                .collect_tuple()
                .unwrap();
            let pos = TttPosition::from_bitboards(board_x, board_o, e.pos.get_turn());

            let probs = e
                .probs
                .iter()
                .map(|(m, p)| (TttMove::from_idx(transform_sq(m.to_idx())), *p))
                .collect_vec();

            let winner = e.winner;
            DataEntry { pos, probs, winner }
        };

        let rows_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = TttGame::BOARD_SIZE - 1 - r;
                let ct = c;
                rt * TttGame::BOARD_SIZE + ct
            })
        };
        let columns_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = r;
                let ct = TttGame::BOARD_SIZE - 1 - c;
                rt * TttGame::BOARD_SIZE + ct
            })
        };
        let diagonal_mirror = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / TttGame::BOARD_SIZE, idx % TttGame::BOARD_SIZE);
                let rt = c;
                let ct = r;
                rt * TttGame::BOARD_SIZE + ct
            })
        };

        /*
         * Use all combination of the basic transforms:
         * original
         * rows mirror
         * columns mirror
         * diagonal mirror
         * rows + columns = rotate 180
         * rows + diagonal = rotate 90
         * columns + diagonal = rotate 90 (other direction)
         * row + columns + diagonal = other diagonal mirror
         */
        let mut entries = vec![entry];
        entries.extend(entries.iter().map(rows_mirror).collect_vec());
        entries.extend(entries.iter().map(columns_mirror).collect_vec());
        entries.extend(entries.iter().map(diagonal_mirror).collect_vec());
        entries
    }
}

impl<const BOARD_SIZE: usize> SelfPlayGame for HexGame<BOARD_SIZE> {
    fn produce_transformed_data_entries(entry: DataEntry<Self>) -> Vec<DataEntry<Self>> {
        let transform = |e: &DataEntry<Self>, transform_sq: &dyn Fn(usize) -> usize| {
            let (board_red, board_blue) = [e.pos.board_red, e.pos.board_blue]
                .iter()
                .map(|b| {
                    let mut bt = HexBitboard::new();
                    for idx in 0..BOARD_SIZE * BOARD_SIZE {
                        bt.set(transform_sq(idx), b.get(idx));
                    }
                    bt
                })
                .collect_tuple()
                .unwrap();
            let pos = HexPosition::new_from_board(board_red, board_blue, e.pos.get_turn());

            let probs = e
                .probs
                .iter()
                .map(|(m, p)| (HexMove::from_idx(transform_sq(m.to_idx())), *p))
                .collect_vec();

            let winner = e.winner;
            DataEntry { pos, probs, winner }
        };

        let rotate_180 = |e: &DataEntry<Self>| {
            transform(e, &|idx| {
                let (r, c) = (idx / BOARD_SIZE, idx % BOARD_SIZE);
                let rt = BOARD_SIZE - 1 - r;
                let ct = BOARD_SIZE - 1 - c;
                rt * BOARD_SIZE + ct
            })
        };

        let mut entries = vec![entry];
        entries.extend(entries.iter().map(rotate_180).collect_vec());
        entries
    }
}

impl SelfPlayGame for ChessGame {
    fn produce_transformed_data_entries(entry: DataEntry<Self>) -> Vec<DataEntry<Self>> {
        let transform = |e: &DataEntry<Self>,
                         transform_sq: &dyn Fn(
            cattus::chess::chess::Square,
        ) -> cattus::chess::chess::Square| {
            let b = &e.pos.board;
            let pieces = b
                .combined()
                .into_iter()
                .map(|square| {
                    (
                        transform_sq(square),
                        b.piece_on(square).unwrap(),
                        b.color_on(square).unwrap(),
                    )
                })
                .collect_vec();

            let board =
                cattus::chess::chess::Board::try_from(cattus::chess::chess::BoardBuilder::setup(
                    pieces.iter(),
                    b.side_to_move(),
                    b.castle_rights(cattus::chess::chess::Color::White),
                    b.castle_rights(cattus::chess::chess::Color::Black),
                    b.en_passant().map(|square| transform_sq(square).get_file()),
                ))
                .expect("unable to transform board");
            let pos = ChessPosition {
                board,
                fifty_rule_count: e.pos.fifty_rule_count,
            };

            let probs = e
                .probs
                .iter()
                .map(|(m, p)| {
                    (
                        ChessMove::new(cattus::chess::chess::ChessMove::new(
                            transform_sq(m.as_ref().get_source()),
                            transform_sq(m.as_ref().get_dest()),
                            m.as_ref().get_promotion(),
                        )),
                        *p,
                    )
                })
                .collect_vec();

            let winner = e.winner;
            DataEntry { pos, probs, winner }
        };

        let rows_mirror = |e: &DataEntry<Self>| {
            transform(e, &|sq| {
                cattus::chess::chess::Square::make_square(
                    cattus::chess::chess::Rank::from_index(
                        ChessGame::BOARD_SIZE - 1 - sq.get_rank().to_index(),
                    ),
                    sq.get_file(),
                )
            })
        };
        let columns_mirror = |e: &DataEntry<Self>| {
            transform(e, &|sq| {
                cattus::chess::chess::Square::make_square(
                    sq.get_rank(),
                    cattus::chess::chess::File::from_index(
                        ChessGame::BOARD_SIZE - 1 - sq.get_file().to_index(),
                    ),
                )
            })
        };
        let diagonal_mirror = |e: &DataEntry<Self>| {
            transform(e, &|sq| {
                cattus::chess::chess::Square::make_square(
                    cattus::chess::chess::Rank::from_index(sq.get_file().to_index()),
                    cattus::chess::chess::File::from_index(sq.get_rank().to_index()),
                )
            })
        };

        let b = &entry.pos.board;
        let has_castle_rights = b.castle_rights(cattus::chess::chess::Color::White)
            != cattus::chess::chess::CastleRights::NoRights
            || b.castle_rights(cattus::chess::chess::Color::Black)
                != cattus::chess::chess::CastleRights::NoRights;
        let has_pawns = b.pieces(cattus::chess::chess::Piece::Pawn).0 != 0;

        /*
         * Use all combination of the basic transforms:
         * original
         * rows mirror
         * columns mirror
         * diagonal mirror
         * rows + columns = rotate 180
         * rows + diagonal = rotate 90
         * columns + diagonal = rotate 90 (other direction)
         * row + columns + diagonal = other diagonal mirror
         */
        let mut entries = vec![entry];
        if !has_castle_rights {
            entries.extend(entries.iter().map(columns_mirror).collect_vec());
        }
        if !has_castle_rights && !has_pawns {
            entries.extend(entries.iter().map(rows_mirror).collect_vec());
            entries.extend(entries.iter().map(diagonal_mirror).collect_vec());
        }
        entries
    }
}
