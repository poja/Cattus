use crate::game_utils::train::Encoder;
use crate::game_utils::{game, train};
use crate::hex::hex_game;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

pub struct SimpleEncoder {}

impl SimpleEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl train::Encoder<hex_game::HexGame> for SimpleEncoder {
    fn encode_moves(&self, moves: &Vec<(hex_game::Location, f32)>) -> Vec<f32> {
        return vec![];
    }
    fn decode_moves(&self, moves: &Vec<f32>) -> Vec<(hex_game::Location, f32)> {
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

pub struct SimpleNetwork {}

impl SimpleNetwork {
    pub fn new() -> Self {
        Self {}
    }

    pub fn make_game_prediction(&self, position: &hex_game::HexPosition) -> f32 {
        let encoder = SimpleEncoder::new();
        let encoded_position = encoder.encode_position(position);
        let tensor: Tensor<f32> = Tensor::new(&[1, 121])
            .with_values(&encoded_position)
            .expect("Can't create tensor");

        // In this file test_in_input is being used while in the python script,
        // that generates the saved model from Keras model it has a name "test_in".
        // For multiple inputs _input is not being appended to signature input parameter name.
        let signature_input_parameter_name = "test_in_input";
        let signature_output_parameter_name = "test_out";

        // Initialize save_dir, input tensor, and an empty graph
        let save_dir = "C:/code/rl/model/m1";
        let mut graph = Graph::new();

        // Load saved model bundle (session state + meta_graph data)
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
                .expect("Can't load saved model");

        // Get the session from the loaded model bundle
        let session = &bundle.session;

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

        // Manages inputs and outputs for the execution of the graph
        let mut args = SessionRunArgs::new();
        args.add_feed(&input_op, 0, &tensor); // Add any inputs

        let out = args.request_fetch(&output_op, 0); // Request outputs

        // Run model
        session
            .run(&mut args) // Pass to session to run
            .expect("Error occurred during calculations");

        // Fetch outputs after graph execution
        let out_res: f32 = args.fetch(out).unwrap()[0];
        println!("Results: {:?}", out_res);
        return out_res;
    }
}
