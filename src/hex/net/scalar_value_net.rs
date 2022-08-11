use crate::game_utils::game::{self, GamePosition, IGame};
use crate::game_utils::mcts::ValueFunction;
use crate::game_utils::self_play::Encoder;
use crate::hex::hex_game::{self, HexGame, HexPosition};
use crate::hex::net::encoder::SimpleEncoder;
use itertools::Itertools;
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct ScalarValNet {
    bundle: SavedModelBundle,
    encoder: SimpleEncoder,
    input_op: Operation,
    output_op: Operation,
}

impl ScalarValNet {
    pub fn new(model_path: String) -> Self {
        // In this file in_position_input is being used while in the python script,
        // that generates the saved model from Keras model it has a name "in_position".
        // For multiple inputs _input is not being appended to signature input parameter name.
        let signature_input_parameter_name = "in_position_input";
        let signature_output_parameter_name = "out_value";

        // Load saved model bundle (session state + meta_graph data)
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, &model_path)
                .expect("Can't load saved model");

        // Get signature metadata from the model bundle
        let signature = bundle
            .meta_graph_def()
            .get_signature("serving_default")
            .unwrap();

        // Get input/output info
        let input_info = signature.get_input(signature_input_parameter_name).unwrap();
        let output_info = signature
            .get_output(signature_output_parameter_name)
            .unwrap();

        // Get input/output ops from graph
        let input_op = graph
            .operation_by_name_required(&input_info.name().name)
            .unwrap();
        let output_op = graph
            .operation_by_name_required(&output_info.name().name)
            .unwrap();

        Self {
            bundle: bundle,
            encoder: SimpleEncoder::new(),
            input_op: input_op,
            output_op: output_op,
        }
    }

    pub fn evaluate_position(
        &self,
        position: &hex_game::HexPosition,
    ) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        if position.get_turn() == game::GameColor::Player1 {
            return self.evaluate_position_impl(position);
        } else {
            let flipped_pos = hex_game::HexPosition::flip_of(position);
            let res = self.evaluate_position_impl(&flipped_pos);
            /* Flip scalar value */
            return (-res.0, res.1);
        }
    }

    fn evaluate_position_impl(
        &self,
        position: &hex_game::HexPosition,
    ) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        let encoded_position = self.encoder.encode_position(position);
        let input: Tensor<f32> = Tensor::new(&[1, 121])
            .with_values(&encoded_position)
            .expect("Can't create input tensor");

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.input_op, 0, &input);
        let output = args.request_fetch(&self.output_op, 0);

        self.bundle
            .session
            .run(&mut args)
            .expect("Error occurred during calculations");

        let val = args.fetch(output).unwrap()[0];

        /* We don't have anything smart to say per move */
        /* Assign uniform probabilities to all legal moves */
        let moves = position.get_legal_moves();
        let move_prob = 1.0 / moves.len() as f32;
        let moves_probs = moves.iter().map(|m| (*m, move_prob)).collect_vec();

        return (val, moves_probs);
    }
}

impl ValueFunction<HexGame> for ScalarValNet {
    fn evaluate(&mut self, position: &HexPosition) -> (f32, Vec<(<HexGame as IGame>::Move, f32)>) {
        return self.evaluate_position(position);
    }
}
