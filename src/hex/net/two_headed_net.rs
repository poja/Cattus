use crate::game::common::{GamePosition, IGame};
use crate::game::encoder::Encoder;
use crate::game::mcts::ValueFunction;
use crate::hex::hex_game::{self, HexGame, HexPosition};
use crate::hex::net::common;
use crate::hex::net::encoder::SimpleEncoder;
use itertools::Itertools;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct TwoHeadedNet {
    bundle: SavedModelBundle,
    encoder: SimpleEncoder,
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
            encoder: SimpleEncoder::new(),
            input_op: input_op,
            output_scalar_op: output_scalar_op,
            output_probs_op: output_probs_op,
        }
    }

    pub fn evaluate_position(
        &self,
        position: &hex_game::HexPosition,
    ) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        let (flipped_pos, is_flipped) = common::flip_pos_if_needed(*position);
        let eval = self.evaluate_position_impl(&flipped_pos);
        return common::flip_score_if_needed(eval, is_flipped);
    }

    fn evaluate_position_impl(
        &self,
        position: &hex_game::HexPosition,
    ) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        let encoded_position = self.encoder.encode_position(position);
        let input_dim = hex_game::BOARD_SIZE * hex_game::BOARD_SIZE;
        let input: Tensor<f32> = Tensor::new(&[1, input_dim as u64])
            .with_values(&encoded_position)
            .expect("Can't create input tensor");

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output_scalar = args.request_fetch(&self.output_scalar_op, 1);
        let output_probs = args.request_fetch(&self.output_probs_op, 0);

        self.bundle
            .session
            .run(&mut args)
            .expect("Error occurred during calculations");

        let val: f32 = args.fetch(output_scalar).unwrap()[0];

        let probs: Tensor<f32> = args.fetch(output_probs).unwrap();
        for idx in 0..input_dim {
            assert!(probs[idx] >= 0.0);
        }
        let moves = position.get_legal_moves();
        let moves_probs = moves
            .iter()
            .map(|(r, c)| ((*r, *c), probs[r * hex_game::BOARD_SIZE + c]))
            .collect_vec();

        return (val, moves_probs);
    }
}

impl ValueFunction<HexGame> for TwoHeadedNet {
    fn evaluate(&mut self, position: &HexPosition) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        return self.evaluate_position(position);
    }
}
