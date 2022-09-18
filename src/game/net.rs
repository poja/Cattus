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

    pub fn softmax_normalizatione(x: Vec<f32>) -> Vec<f32> {
        let max_p = x.iter().cloned().fold(f32::MIN, f32::max);
        let x = x.iter().map(|p| (p - max_p).exp()).collect_vec();
        let p_sum: f32 = x.iter().sum();
        return x.iter().map(|p| p / p_sum).collect_vec();
    }
}
