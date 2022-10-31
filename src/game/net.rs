use crate::game::cache::ValueFuncCache;
use crate::game::common::{GameBitboard, GameColor, GameMove, GamePosition, IGame};
use itertools::Itertools;
use std::sync::Arc;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct TwoHeadedNetBase<Game: IGame, const CPU: bool> {
    bundle: SavedModelBundle,
    input_op: Operation,
    value_head: Operation,
    policy_head: Operation,
    cache: Option<Arc<ValueFuncCache<Game>>>,
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
        }
    }

    pub fn run_net(&self, input: Tensor<f32>) -> (f32, Vec<f32>) {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output_scalar = args.request_fetch(&self.value_head, 1);
        let output_moves_scores = args.request_fetch(&self.policy_head, 0);

        self.bundle
            .session
            .run(&mut args)
            .expect("Error occurred during calculations");

        let mut val: f32 = args.fetch(output_scalar).unwrap()[0];
        if val.is_nan() {
            val = 0.0;
        }

        let moves_scores: Tensor<f32> = args.fetch(output_moves_scores).unwrap();
        let moves_scores = moves_scores
            .iter()
            .map(|s| if s.is_nan() { f32::MIN } else { *s })
            .collect_vec();

        (val, moves_scores)
    }

    pub fn evaluate(
        &mut self,
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
        let planes = to_planes(pos);
        let input = planes_to_tensor::<Game, CPU>(planes);
        let (val, move_scores) = self.run_net(input);

        let moves = pos.get_legal_moves();
        let moves_probs = calc_moves_probs::<Game>(moves, &move_scores);

        (val, moves_probs)
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

pub fn planes_to_tensor<Game: IGame, const CPU: bool>(planes: Vec<Game::Bitboard>) -> Tensor<f32> {
    let planes_num = planes.len();

    let mut encoded_position = vec![0.0; planes_num * Game::BOARD_SIZE * Game::BOARD_SIZE];
    for (plane_idx, plane) in planes.into_iter().enumerate() {
        for square in 0..(Game::BOARD_SIZE * Game::BOARD_SIZE) {
            let idx = if CPU {
                square * planes_num + plane_idx
            } else {
                plane_idx * Game::BOARD_SIZE * Game::BOARD_SIZE + square
            };
            encoded_position[idx] = match plane.get(square) {
                true => 1.0,
                false => 0.0,
            };
        }
    }

    let dims = if CPU {
        [
            1,
            Game::BOARD_SIZE as u64,
            Game::BOARD_SIZE as u64,
            planes_num as u64,
        ]
    } else {
        [
            1,
            planes_num as u64,
            Game::BOARD_SIZE as u64,
            Game::BOARD_SIZE as u64,
        ]
    };
    Tensor::new(&dims)
        .with_values(&encoded_position)
        .expect("Can't create input tensor")
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
