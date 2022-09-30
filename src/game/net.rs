use crate::game::common::{Bitboard, GameColor, GameMove, GamePosition, IGame};
use itertools::Itertools;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct TwoHeadedNetBase {
    bundle: SavedModelBundle,
    input_op: Operation,
    value_head: Operation,
    policy_head: Operation,
}

impl TwoHeadedNetBase {
    pub fn new(model_path: &String) -> Self {
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
            bundle: bundle,
            input_op: input_op,
            value_head: value_head,
            policy_head: policy_head,
        }
    }

    fn run_net(&self, input: Tensor<f32>) -> (f32, Vec<f32>) {
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
            .into_iter()
            .map(|s| if s.is_nan() { f32::MIN } else { *s })
            .collect_vec();

        return (val, moves_scores);
    }

    pub fn evaluate<Game: IGame, B: Bitboard, const BOARD_SIZE: usize>(
        &mut self,
        position: &Game::Position,
        to_planes: impl Fn(&Game::Position) -> Vec<B>,
    ) -> (f32, Vec<(Game::Move, f32)>) {
        let (flipped_pos, is_flipped) = flip_pos_if_needed(*position);

        let planes = to_planes(&flipped_pos);
        let input = planes_to_tensor::<B, BOARD_SIZE>(planes);
        let (val, move_scores) = self.run_net(input);

        let moves = flipped_pos.get_legal_moves();
        let moves_probs = TwoHeadedNetBase::calc_moves_probs(moves, &move_scores);

        return flip_score_if_needed((val, moves_probs), is_flipped);
    }

    pub fn calc_moves_probs<M: GameMove>(moves: Vec<M>, move_scores: &Vec<f32>) -> Vec<(M, f32)> {
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

        return moves.into_iter().zip(probs.into_iter()).collect_vec();
    }
}

pub fn planes_to_tensor<B: Bitboard, const BOARD_SIZE: usize>(planes: Vec<B>) -> Tensor<f32> {
    let cpu = true;
    let planes_num = planes.len();

    let mut encoded_position = vec![0.0; planes_num * BOARD_SIZE * BOARD_SIZE];
    for (plane_idx, plane) in planes.into_iter().enumerate() {
        for square in 0..(BOARD_SIZE * BOARD_SIZE) {
            let idx = if cpu {
                square * planes_num + plane_idx
            } else {
                plane_idx * BOARD_SIZE * BOARD_SIZE + square
            };
            encoded_position[idx] = match plane.get(square) {
                true => 1.0,
                false => 0.0,
            };
        }
    }

    let dims = if cpu {
        [1, BOARD_SIZE as u64, BOARD_SIZE as u64, planes_num as u64]
    } else {
        [1, planes_num as u64, BOARD_SIZE as u64, BOARD_SIZE as u64]
    };
    return Tensor::new(&dims)
        .with_values(&encoded_position)
        .expect("Can't create input tensor");
}

pub fn flip_pos_if_needed<Position: GamePosition>(pos: Position) -> (Position, bool) {
    if pos.get_turn() == GameColor::Player1 {
        return (pos, false);
    } else {
        return (pos.get_flip(), true);
    }
}

pub fn flip_score_if_needed<Move: GameMove>(
    net_res: (f32, Vec<(Move, f32)>),
    pos_flipped: bool,
) -> (f32, Vec<(Move, f32)>) {
    if !pos_flipped {
        return net_res;
    } else {
        let (val, moves_probs) = net_res;

        /* Flip scalar value */
        let val = -val;

        /* Flip moves */
        let moves_probs = moves_probs
            .into_iter()
            .map(|(m, p)| (m.get_flip(), p))
            .collect_vec();

        return (val, moves_probs);
    }
}
