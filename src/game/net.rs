use crate::game::common::{Bitboard, GameMove};
use itertools::Itertools;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct TwoHeadedNetBase {
    bundle: SavedModelBundle,
    input_op: Operation,
    value_head: Operation,
    policy_head: Operation,
}

impl TwoHeadedNetBase {
    pub fn new(
        model_path: &String,
        input_name: &str,
        value_head_name: &str,
        policy_head_name: &str,
    ) -> Self {
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
        let input_info = signature.get_input(input_name).unwrap();
        let output_scalar_info = signature.get_output(value_head_name).unwrap();
        let output_probs_info = signature.get_output(policy_head_name).unwrap();

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

    pub fn run_net(&self, input: Tensor<f32>) -> (f32, Tensor<f32>) {
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output_scalar = args.request_fetch(&self.value_head, 1);
        let output_probs = args.request_fetch(&self.policy_head, 0);

        self.bundle
            .session
            .run(&mut args)
            .expect("Error occurred during calculations");

        let mut val: f32 = args.fetch(output_scalar).unwrap()[0];
        if val.is_nan() {
            val = 0.0;
        }

        let mut probs: Tensor<f32> = args.fetch(output_probs).unwrap();
        for idx in 0..probs.len() {
            if probs[idx].is_nan() {
                probs[idx] = f32::MIN;
            }
        }

        return (val, probs);
    }

    pub fn calc_moves_probs<M: GameMove>(moves: Vec<M>, move_scores: Tensor<f32>) -> Vec<(M, f32)> {
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

    pub fn planes_to_tensor<B: Bitboard>(planes: Vec<B>, board_size: usize) -> Tensor<f32> {
        let cpu = true;
        let planes_num = planes.len();

        let mut encoded_position = vec![0.0; (planes_num * board_size * board_size) as usize];
        for (plane_idx, plane) in planes.into_iter().enumerate() {
            for square in 0..(board_size * board_size) {
                let idx = if cpu {
                    square * planes_num + plane_idx
                } else {
                    plane_idx * board_size * board_size + square
                };
                encoded_position[idx] = match plane.get(square as u8) {
                    true => 1.0,
                    false => 0.0,
                };
            }
        }

        let dims = if cpu {
            [1, board_size as u64, board_size as u64, planes_num as u64]
        } else {
            [1, planes_num as u64, board_size as u64, board_size as u64]
        };
        return Tensor::new(&dims)
            .with_values(&encoded_position)
            .expect("Can't create input tensor");
    }
}
