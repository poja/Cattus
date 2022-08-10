use crate::game_utils::game::GamePosition;
use crate::game_utils::mcts::ValueFunction;
use crate::game_utils::self_play::Encoder;
use crate::game_utils::{game, self_play};
use crate::hex::hex_game::{self, HexGame, HexPosition};
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl self_play::Encoder<hex_game::HexGame> for SimpleEncoder {
    fn encode_moves(&self, _moves: &Vec<(hex_game::Location, f32)>) -> Vec<f32> {
        return vec![];
    }
    fn decode_moves(&self, _moves: &Vec<f32>) -> Vec<(hex_game::Location, f32)> {
        return vec![];
    }
    fn encode_position(&self, position: &hex_game::HexPosition) -> Vec<f32> {
        let mut vec = Vec::new();
        for r in 0..hex_game::BOARD_SIZE {
            for c in 0..hex_game::BOARD_SIZE {
                vec.push(match position.get_tile(r, c) {
                    hex_game::Hexagon::Full(color) => match color {
                        game::GameColor::Player1 => 1.0,
                        game::GameColor::Player2 => -1.0,
                    },
                    hex_game::Hexagon::Empty => 0.0,
                });
            }
        }
        return vec;
    }
}

pub struct SimpleNetwork {
    bundle: SavedModelBundle,
    encoder: SimpleEncoder,
    input_op: Operation,
    output_op: Operation,
}

impl SimpleNetwork {
    pub fn new(model_path: String) -> Self {
        // In this file test_in_input is being used while in the python script,
        // that generates the saved model from Keras model it has a name "test_in".
        // For multiple inputs _input is not being appended to signature input parameter name.
        let signature_input_parameter_name = "test_in_input";
        let signature_output_parameter_name = "test_out";

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

    pub fn evaluate_position(&self, position: &hex_game::HexPosition) -> f32 {
        if position.get_turn() == game::GameColor::Player1 {
            return self.evaluate_position_impl(position);
        } else {
            let flipped_pos = hex_game::HexPosition::flip_of(position);
            return -self.evaluate_position_impl(&flipped_pos);
        }
    }

    fn evaluate_position_impl(&self, position: &hex_game::HexPosition) -> f32 {
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

        return args.fetch(output).unwrap()[0];
    }
}

impl ValueFunction<HexGame> for SimpleNetwork {
    fn evaluate(&mut self, position: &HexPosition) -> f32 {
        return self.evaluate_position(position);
    }
}
