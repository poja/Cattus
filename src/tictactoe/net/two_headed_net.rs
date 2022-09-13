use crate::game::common::{GamePosition, IGame};
use crate::game::mcts::ValueFunction;
use crate::tictactoe::net::common;
use crate::tictactoe::net::encoder::Encoder;
use crate::tictactoe::tictactoe_game::{TicTacToeGame, TicTacToePosition};
use itertools::Itertools;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct TwoHeadedNet {
    bundle: SavedModelBundle,
    encoder: Encoder,
    input_op: Operation,
    output_scalar_op: Operation,
    output_probs_op: Operation,
}

impl TwoHeadedNet {
    pub fn new(model_path: &String) -> Self {
        let signature_input_parameter_name = "in_position";
        let signature_output_value_parameter_name = "out_value";
        let signature_output_probs_parameter_name = "out_probs";

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
        let input_info = signature.get_input(signature_input_parameter_name).unwrap();
        let output_scalar_info = signature
            .get_output(signature_output_value_parameter_name)
            .unwrap();
        let output_probs_info = signature
            .get_output(signature_output_probs_parameter_name)
            .unwrap();

        // Get input/output ops from graph
        let input_op = graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        let output_scalar_op = graph
            .operation_by_name_required(&output_scalar_info.name().name)
            .unwrap();
        let output_probs_op = graph
            .operation_by_name_required(&output_probs_info.name().name)
            .unwrap();

        Self {
            bundle: bundle,
            encoder: Encoder::new(),
            input_op: input_op,
            output_scalar_op: output_scalar_op,
            output_probs_op: output_probs_op,
        }
    }

    pub fn evaluate_position(
        &self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        let (flipped_pos, is_flipped) = common::flip_pos_if_needed(*position);
        let eval = self.evaluate_position_impl(&flipped_pos);
        return common::flip_score_if_needed(eval, is_flipped);
    }

    fn evaluate_position_impl(
        &self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        let input = self.encoder.encode_position(position);

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output_scalar = args.request_fetch(&self.output_scalar_op, 1);
        let output_probs = args.request_fetch(&self.output_probs_op, 0);

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
        let moves = position.get_legal_moves();
        let moves_probs = moves
            .iter()
            .map(|move_| ((*move_), probs[move_.to_idx() as usize]))
            .collect_vec();

        return (val, moves_probs);
    }
}

impl ValueFunction<TicTacToeGame> for TwoHeadedNet {
    fn evaluate(
        &mut self,
        position: &TicTacToePosition,
    ) -> (f32, Vec<(<TicTacToeGame as IGame>::Move, f32)>) {
        return self.evaluate_position(position);
    }
}
