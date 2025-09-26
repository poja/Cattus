use itertools::Itertools;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::Instant;

use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame, PlayerRand};
use crate::util::metrics::RunningAverage;

/// Monte Carlo Tree Search (MCTS) implementation

#[derive(Clone, Copy)]
struct MctsNode<Position: GamePosition> {
    position: Position,
}

impl<Position: GamePosition> MctsNode<Position> {
    pub fn from_position(position: Position) -> Self {
        Self { position }
    }
}

#[derive(Clone, Copy)]
struct MctsEdge<Move: GameMove> {
    m: Move,

    /// The initial score calculated for this node.
    /// In range [0, 1], "probability"
    init_score: f32,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    score_w: f32,
}

impl<Move: GameMove> MctsEdge<Move> {
    pub fn new(m: Move, init_score: f32) -> Self {
        Self {
            m,
            init_score,
            simulations_n: 0,
            score_w: 0.0,
        }
    }
}

pub struct MctsPlayer<Game: IGame> {
    search_tree: DiGraph<MctsNode<Game::Position>, MctsEdge<Game::Move>>,
    root: Option<NodeIndex>,

    sim_num: u32,
    explore_factor: f32,
    temperature: f32,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    value_func: Arc<dyn ValueFunction<Game>>,

    search_duration_metric: RunningAverage,
}

pub struct MctsParams<Game: IGame> {
    pub sim_num: u32,
    pub explore_factor: f32,
    pub prior_noise_alpha: f32,
    pub prior_noise_epsilon: f32,
    pub value_func: Arc<dyn ValueFunction<Game>>,
}
impl<Game: IGame> MctsParams<Game> {
    pub fn new(sim_num: u32, value_func: Arc<dyn ValueFunction<Game>>) -> Self {
        Self {
            sim_num,
            explore_factor: std::f32::consts::SQRT_2,
            prior_noise_alpha: 0.0,
            prior_noise_epsilon: 0.0,
            value_func,
        }
    }
}
impl<Game: IGame> Clone for MctsParams<Game> {
    fn clone(&self) -> Self {
        Self {
            sim_num: self.sim_num,
            explore_factor: self.explore_factor,
            prior_noise_alpha: self.prior_noise_alpha,
            prior_noise_epsilon: self.prior_noise_epsilon,
            value_func: Arc::clone(&self.value_func),
        }
    }
}

impl<Game: IGame> MctsPlayer<Game> {
    pub fn new(params: MctsParams<Game>) -> Self {
        assert!(params.sim_num > 0);
        assert!(params.explore_factor >= 0.0);
        assert!(params.prior_noise_alpha >= 0.0);
        assert!((0.0..=1.0).contains(&params.prior_noise_epsilon));

        let search_duration_metric_name = "mcts.search_duration";
        metrics::describe_gauge!(
            search_duration_metric_name,
            metrics::Unit::Seconds,
            "Duration of MCTS search"
        );
        let search_duration_metric =
            RunningAverage::new(0.99, metrics::gauge!(search_duration_metric_name));

        Self {
            search_tree: DiGraph::new(),
            root: None,
            sim_num: params.sim_num,
            explore_factor: params.explore_factor,
            prior_noise_alpha: params.prior_noise_alpha,
            prior_noise_epsilon: params.prior_noise_epsilon,
            temperature: 1.0,
            value_func: params.value_func,
            search_duration_metric,
        }
    }

    fn detect_repetition(&self, pos_history: &[Game::Position], trajectory: &[EdgeIndex]) -> bool {
        let repetition_limit = match Game::REPETITION_LIMIT {
            Some(l) if l > 1 => l,
            _ => return false,
        };

        let trajectory = trajectory.iter().map(|idx| {
            let (_e_source, e_target) = self.search_tree.edge_endpoints(*idx).unwrap();
            &self.search_tree[e_target].position
        });
        let full_trajectory = pos_history.iter().chain(trajectory);

        let mut repetitions = HashMap::new();
        for pos in full_trajectory {
            let repeat = repetitions.entry(pos).or_insert(0);
            *repeat += 1;
            if *repeat >= repetition_limit {
                return true;
            }
        }
        false
    }

    fn develop_tree(&mut self, pos_history: &[Game::Position]) {
        assert!(self.sim_num > 1);
        for _ in 0..self.sim_num {
            /* Select a leaf node */
            let path_to_selection = self.select();

            let repetition_reached = self.detect_repetition(pos_history, &path_to_selection);
            let leaf_id: NodeIndex = if path_to_selection.is_empty() {
                self.root.unwrap()
            } else {
                let (_e_source, e_target) = self
                    .search_tree
                    .edge_endpoints(*path_to_selection.last().unwrap())
                    .unwrap();
                e_target
            };
            let leaf_pos = &self.search_tree[leaf_id].position;

            let eval = if repetition_reached {
                0.0
            } else if leaf_pos.is_over() {
                GameColor::to_idx(leaf_pos.get_winner()) as f32
            } else {
                /* Run value function once to obtain "simulation" value and initial children scores (probabilities) */
                let (per_move_val, eval) = self.simulate(leaf_id);

                /* Expand leaf and assign initial scores */
                self.create_children(leaf_id, per_move_val);

                /* Add Dirichlet noise to root initial probabilities */
                if leaf_id == self.root.unwrap() {
                    self.add_dirichlet_noise(leaf_id);
                }

                eval
            };

            /* back propagate the position score to the parents */
            self.backpropagate(path_to_selection, eval);
        }
    }

    /* Return path to selected leaf node */
    fn select(&self) -> Vec<EdgeIndex> {
        let mut path: Vec<EdgeIndex> = vec![];

        let mut node_id = self.root.unwrap();
        loop {
            let node = &self.search_tree[node_id];

            /* Node is leaf, done */
            if node.position.is_over() || self.search_tree.edges(node_id).next().is_none() {
                return path;
            }

            let node_simcount = 1 + self
                .search_tree
                .edges(node_id)
                .map(|edge| edge.weight().simulations_n)
                .sum::<u32>();

            /* Node is not a leaf, choose best child and continue in it's sub tree */
            let edge = self
                .search_tree
                .edges(node_id)
                .max_by(|e1, e2| {
                    let val1 = self.calc_selection_heuristic(e1.weight(), node_simcount);
                    let val2 = self.calc_selection_heuristic(e2.weight(), node_simcount);
                    val1.partial_cmp(&val2).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            path.push(edge.id());
            node_id = edge.target();
        }
    }

    fn calc_selection_heuristic(&self, edge: &MctsEdge<Game::Move>, parent_simcount: u32) -> f32 {
        let exploit = if edge.simulations_n == 0 {
            0.0
        } else {
            edge.score_w / edge.simulations_n as f32
        };

        let explore = self.explore_factor
            * edge.init_score
            * ((parent_simcount as f32).sqrt() / (1 + edge.simulations_n) as f32);

        exploit + explore
    }

    fn create_children(
        &mut self,
        parent_id: NodeIndex,
        per_move_init_score: Vec<(Game::Move, f32)>,
    ) {
        let parent_pos = self.search_tree[parent_id].position;
        assert!(!parent_pos.is_over());

        debug_assert!({
            let moves_actual: HashSet<Game::Move> =
                HashSet::from_iter(per_move_init_score.iter().map(|(m, _p)| *m));
            let moves_expected: HashSet<Game::Move> =
                HashSet::from_iter(parent_pos.get_legal_moves().iter().cloned());
            moves_actual == moves_expected
        });

        for (m, p) in per_move_init_score {
            let leaf_pos = parent_pos.get_moved_position(m);
            let leaf_id = self.search_tree.add_node(MctsNode::from_position(leaf_pos));
            self.search_tree
                .add_edge(parent_id, leaf_id, MctsEdge::new(m, p));
        }
    }

    fn simulate(&mut self, leaf_id: NodeIndex) -> (Vec<(Game::Move, f32)>, f32) {
        let position = &self.search_tree[leaf_id].position;
        assert!(!position.is_over());
        self.value_func.evaluate(position)
    }

    fn backpropagate(&mut self, path: Vec<EdgeIndex>, score: f32) {
        for edge_id in path {
            let (e_source, _e_target) = self.search_tree.edge_endpoints(edge_id).unwrap();
            let player_to_play = self.search_tree[e_source].position.get_turn();
            let edge = self.search_tree.edge_weight_mut(edge_id).unwrap();
            edge.simulations_n += 1;
            edge.score_w += match player_to_play {
                GameColor::Player1 => score,
                GameColor::Player2 => -score,
            };
        }
    }

    fn find_node_with_position(
        &self,
        position: &Game::Position,
        depth_limit: u32,
    ) -> Option<NodeIndex> {
        let mut layer = vec![self.root.unwrap()];

        for _ in 0..depth_limit {
            let mut next_layer = Vec::new();

            for node in layer {
                if &self.search_tree[node].position == position {
                    return Some(node);
                }
                // Add children to next layer
                for edge in self.search_tree.edges(node) {
                    next_layer.push(edge.target())
                }
            }
            layer = next_layer;
        }
        None
    }

    fn remove_all_but_subtree(&mut self, sub_tree_root: NodeIndex) {
        if self.root.unwrap() == sub_tree_root {
            return;
        }

        // In petgraph, when you remove a node, the indices of the other nodes
        // change. So instead of removing nodes from the current tree, we copy
        // the sub tree to a new graph
        let mut new_tree = DiGraph::new();
        let new_root = new_tree.add_node(self.search_tree[sub_tree_root]);
        let mut nodes = vec![(sub_tree_root, new_root)];

        while let Some((parent_old, parent_new)) = nodes.pop() {
            for edge in self.search_tree.edges(parent_old) {
                let child_old = edge.target();
                let child_data = &self.search_tree[child_old];
                let child_new = new_tree.add_node(*child_data);
                new_tree.add_edge(parent_new, child_new, *edge.weight());

                nodes.push((child_old, child_new));
            }
        }

        self.search_tree = new_tree;
        self.root = Some(new_root);

        /* If the prior probabilities of the new root were already calculated, add a Dirichlet noise */
        if self.search_tree.edges(new_root).next().is_some() {
            self.add_dirichlet_noise(new_root);
        }
    }

    pub fn calc_moves_probabilities(
        &mut self,
        pos_history: &[Game::Position],
    ) -> Vec<(Game::Move, f32)> {
        let search_start_time = Instant::now();
        let position = pos_history.last().unwrap();

        if self.root.is_some() {
            // Tree was saved from the last search
            // Look for the position in the first three layers of the tree
            // TODO consider increasing depth limit
            match self.find_node_with_position(position, 3) {
                Some(node) => {
                    self.remove_all_but_subtree(node);
                }
                None => {
                    self.search_tree.clear();
                    self.root = None;
                }
            }
        }

        if self.root.is_none() {
            // Init search tree with one root node
            let root = MctsNode::from_position(*position);
            self.root = Some(self.search_tree.add_node(root));
        }
        assert!(position == &self.search_tree[self.root.unwrap()].position);

        // Run all simulations
        self.develop_tree(pos_history);

        // create moves vector (move, sim_count)
        let moves_and_simcounts = self
            .search_tree
            .edges(self.root.unwrap())
            .map(|edge| {
                let e = edge.weight();
                (e.m, e.simulations_n)
            })
            .collect_vec();

        // normalize sim counts to create a valid distribution -> (move, simcount / simcount_total)
        let simcount_total: u32 = moves_and_simcounts
            .iter()
            .map(|&(_, simcount)| simcount)
            .sum();
        let res = moves_and_simcounts
            .into_iter()
            .map(|(m, simcount)| (m, simcount as f32 / simcount_total as f32))
            .collect_vec();

        self.search_duration_metric
            .set(search_start_time.elapsed().as_secs_f64());

        res
    }

    pub fn choose_move_from_probabilities(
        &self,
        moves_probs: &[(Game::Move, f32)],
    ) -> Option<Game::Move> {
        if moves_probs.is_empty() {
            return None;
        }

        if self.temperature == 0.0 {
            let (m, _p) = moves_probs
                .iter()
                .max_by(|(_m1, p1), (_m2, p2)| p1.total_cmp(p2))
                .unwrap();
            Some(*m)
        } else {
            /* prob -> prob^temperature */
            assert!(self.temperature > 0.0);
            let probabilities = moves_probs
                .iter()
                .map(|(_m, p)| p.powf(1.0 / self.temperature))
                .collect_vec();

            /* normalize, prob -> prob / (probs sum) */
            let probs_sum: f32 = probabilities.iter().sum();
            let probabilities = probabilities.iter().map(|p| p / probs_sum).collect_vec();
            let distribution = WeightedIndex::new(probabilities).unwrap();
            Some(moves_probs[distribution.sample(&mut rand::rng())].0)
        }
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        assert!(temperature >= 0.0);
        self.temperature = temperature;
    }

    fn add_dirichlet_noise(&mut self, node_id: NodeIndex) {
        if self.prior_noise_alpha == 0.0 || self.prior_noise_epsilon == 0.0 {
            return;
        }

        assert!((0.0..=1.0).contains(&self.prior_noise_epsilon));

        let moves = self
            .search_tree
            .edges(node_id)
            .map(|e| e.id())
            .collect_vec();
        if moves.len() < 2 {
            return;
        }

        /* The Dirichlet implementation seems to return NaNs and INFs sometimes. */
        /* Keep drawing random noises until valid values are achieved */
        let dist =
            crate::util::dirichlet::Dirichlet::new(&vec![self.prior_noise_alpha; moves.len()])
                .unwrap();
        let noise_vec = loop {
            let noise_vec = dist.sample(&mut rand::rng());
            if noise_vec.iter().all(|n| n.is_finite()) {
                break noise_vec;
            }
        };

        for (edge_id, noise) in moves.into_iter().zip(noise_vec.into_iter()) {
            let m = self.search_tree.edge_weight_mut(edge_id).unwrap();
            m.init_score =
                (1.0 - self.prior_noise_epsilon) * m.init_score + self.prior_noise_epsilon * noise;
            assert!(m.init_score.is_finite());
        }
    }
}

impl<Game: IGame> GamePlayer<Game> for MctsPlayer<Game> {
    fn next_move(&mut self, pos_history: &[Game::Position]) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(pos_history);
        self.choose_move_from_probabilities(&moves)
    }
}

pub trait ValueFunction<Game: IGame>: Sync + Send {
    /// Evaluate a position
    ///
    /// position - The position to evaluate
    ///
    /// Returns a tuple of a scalar value score of the position and per-move scores/probabilities.
    /// The scalar is the current position value in range [-1,1]. 1 if player1 is winning and -1 if player2 is winning
    /// The per-move probabilities should have a sum of 1, greater value is a better move
    fn evaluate(&self, position: &Game::Position) -> (Vec<(Game::Move, f32)>, f32);
}

pub struct ValueFunctionRand;
impl<Game: IGame> ValueFunction<Game> for ValueFunctionRand {
    fn evaluate(&self, position: &Game::Position) -> (Vec<(Game::Move, f32)>, f32) {
        let winner = if position.is_over() {
            position.get_winner()
        } else {
            // Play randomly and return the simulation game result
            let mut player1 = PlayerRand::new();
            let mut player2 = PlayerRand::new();
            let mut game = Game::new_from_pos(*position);

            let (_final_pos, winner) = game.play_until_over(&mut player1, &mut player2);
            winner
        };
        let val = match winner {
            Some(color) => {
                if color == position.get_turn() {
                    1.0
                } else {
                    -1.0
                }
            }
            None => 0.0,
        };

        /* We don't have anything smart to say per move */
        /* Assign uniform probabilities to all legal moves */
        let moves = position.get_legal_moves();
        let move_prob = 1.0 / moves.len() as f32;
        let moves_probs = moves.iter().map(|m| (*m, move_prob)).collect_vec();

        (moves_probs, val)
    }
}
