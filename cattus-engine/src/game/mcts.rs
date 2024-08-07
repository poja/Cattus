use itertools::Itertools;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_distr::Dirichlet;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::game::common::{GameColor, GameMove, GamePlayer, GamePosition, IGame, PlayerRand};
use crate::game::utils::Callback;

/// Monte Carlo Tree Search (MCTS) implementation

#[derive(Clone, Copy)]
struct MCTSNode<Position: GamePosition> {
    position: Position,
}

impl<Position: GamePosition> MCTSNode<Position> {
    pub fn from_position(position: Position) -> Self {
        Self { position }
    }
}

#[derive(Clone, Copy)]
struct MCTSEdge<Move: GameMove> {
    m: Move,

    /// The initial score calculated for this node.
    /// In range [0, 1], "probability"
    init_score: f32,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    score_w: f32,
}

impl<Move: GameMove> MCTSEdge<Move> {
    pub fn new(m: Move, init_score: f32) -> Self {
        Self {
            m,
            init_score,
            simulations_n: 0,
            score_w: 0.0,
        }
    }
}

pub type MctsSearchDurationCallback = Box<dyn Callback<Duration> + Sync + Send>;

pub struct MCTSPlayer<Game: IGame> {
    search_tree: DiGraph<MCTSNode<Game::Position>, MCTSEdge<Game::Move>>,
    root: Option<NodeIndex>,

    sim_num: u32,
    explore_factor: f32,
    temperature: f32,
    prior_noise_alpha: f32,
    prior_noise_epsilon: f32,
    value_func: Arc<dyn ValueFunction<Game>>,

    search_duration_callback: Option<MctsSearchDurationCallback>,
}

impl<Game: IGame> MCTSPlayer<Game> {
    pub fn new(sim_num: u32, value_func: Arc<dyn ValueFunction<Game>>) -> Self {
        Self::new_custom(sim_num, std::f32::consts::SQRT_2, 0.0, 0.0, value_func)
    }

    pub fn new_custom(
        sim_num: u32,
        explore_factor: f32,
        prior_noise_alpha: f32,
        prior_noise_epsilon: f32,
        value_func: Arc<dyn ValueFunction<Game>>,
    ) -> Self {
        assert!(sim_num > 0);
        assert!(explore_factor >= 0.0);
        assert!(prior_noise_alpha >= 0.0);
        assert!((0.0..=1.0).contains(&prior_noise_epsilon));
        Self {
            search_tree: DiGraph::new(),
            root: None,
            sim_num,
            explore_factor,
            prior_noise_alpha,
            prior_noise_epsilon,
            temperature: 1.0,
            value_func,
            search_duration_callback: None,
        }
    }

    fn detect_repetition(&self, trajectory: &[EdgeIndex]) -> bool {
        let repetition_limit = Game::get_repetition_limit();
        if repetition_limit.is_none() || trajectory.is_empty() {
            return false;
        }

        let mut positions = Vec::with_capacity(1 + trajectory.len());

        let (e0_source, _e0_target) = self.search_tree.edge_endpoints(trajectory[0]).unwrap();
        assert!(e0_source == self.root.unwrap());
        positions.push(&self.search_tree[e0_source].position);

        positions.extend(trajectory.iter().map(|idx| {
            let (_e_source, e_target) = self.search_tree.edge_endpoints(*idx).unwrap();
            &self.search_tree[e_target].position
        }));

        let mut repetitions = HashMap::new();
        for pos in positions {
            *repetitions.entry(pos).or_insert(0) += 1;
        }

        for (_pos, repeat) in repetitions {
            if repeat >= repetition_limit.unwrap() {
                return true;
            }
        }
        false
    }

    fn develop_tree(&mut self) {
        for _ in 0..self.sim_num {
            /* Select a leaf node */
            let path_to_selection = self.select();

            let repetition_reached = self.detect_repetition(&path_to_selection);
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

    fn calc_selection_heuristic(&self, edge: &MCTSEdge<Game::Move>, parent_simcount: u32) -> f32 {
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
            let leaf_id = self.search_tree.add_node(MCTSNode::from_position(leaf_pos));
            self.search_tree
                .add_edge(parent_id, leaf_id, MCTSEdge::new(m, p));
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
        position: &Game::Position,
    ) -> Vec<(Game::Move, f32)> {
        let search_start_time = Instant::now();

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
            let root = MCTSNode::from_position(*position);
            self.root = Some(self.search_tree.add_node(root));
        }
        assert!(position == &self.search_tree[self.root.unwrap()].position);

        // Run all simulations
        self.develop_tree();

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

        if let Some(callback) = &self.search_duration_callback {
            callback.call(search_start_time.elapsed());
        }

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
            Some(moves_probs[distribution.sample(&mut rand::thread_rng())].0)
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
        let dist = Dirichlet::new_with_size(self.prior_noise_alpha, moves.len()).unwrap();
        let mut noise_vec;
        loop {
            noise_vec = dist.sample(&mut rand::thread_rng());
            if noise_vec.iter().all(|n| n.is_finite()) {
                break;
            }
        }

        for (edge_id, noise) in moves.into_iter().zip(noise_vec.into_iter()) {
            let m = self.search_tree.edge_weight_mut(edge_id).unwrap();
            m.init_score =
                (1.0 - self.prior_noise_epsilon) * m.init_score + self.prior_noise_epsilon * noise;
            assert!(m.init_score.is_finite());
        }
    }

    pub fn set_search_duration_callback(&mut self, callback: Option<MctsSearchDurationCallback>) {
        self.search_duration_callback = callback
    }
}

impl<Game: IGame> GamePlayer<Game> for MCTSPlayer<Game> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(position);
        self.choose_move_from_probabilities(&moves)
    }
}

pub struct NetStatistics {
    pub activation_count: Option<usize>,
    pub run_duration_average: Option<Duration>,
    pub batch_size_average: Option<f32>,
}
impl NetStatistics {
    pub fn empty() -> Self {
        Self {
            activation_count: None,
            run_duration_average: None,
            batch_size_average: None,
        }
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

    fn get_statistics(&self) -> NetStatistics;
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

    fn get_statistics(&self) -> NetStatistics {
        NetStatistics::empty()
    }
}
