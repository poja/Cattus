use itertools::Itertools;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use crate::game::common::{GameColor, GamePlayer, GamePosition, IGame, PlayerRand};

/// Monte Carlo Tree Search (MCTS) implementation

#[derive(Clone, Copy)]
struct MCTSNode<Position: GamePosition> {
    position: Position,

    /// The initial score calculated for this node.
    /// In range [0, 1], "probability"
    init_score: f32,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    /// Float because could be half for games with ties
    score_w: f32,
}

impl<Position: GamePosition> MCTSNode<Position> {
    pub fn from_position(pos: Position, init_score: f32) -> Self {
        Self {
            position: pos,
            init_score: init_score,
            simulations_n: 0,
            score_w: 0.0,
        }
    }
}

pub struct MCTSPlayer<Game: IGame> {
    search_tree: DiGraph<MCTSNode<Game::Position>, Game::Move>,
    root: Option<NodeIndex>,
    exploration_param_c: f32,
    simulations_per_move: u32,
    temperature: f32,
    value_func: Box<dyn ValueFunction<Game>>,
}

impl<Game: IGame> MCTSPlayer<Game> {
    pub fn new_custom(
        simulations_per_move: u32,
        exploration_param_c: f32,
        value_func: Box<dyn ValueFunction<Game>>,
    ) -> Self {
        Self {
            search_tree: DiGraph::new(),
            root: None,
            exploration_param_c: exploration_param_c,
            simulations_per_move: simulations_per_move,
            temperature: 1.0,
            value_func: value_func,
        }
    }

    fn detect_repetition(&self, trajectory: &Vec<NodeIndex>) -> bool {
        let repetition_limit = Game::get_repetition_limit();
        if repetition_limit.is_none() {
            return false;
        }

        let positions = trajectory
            .iter()
            .map(|idx| &self.search_tree.node_weight(*idx).unwrap().position);

        let mut repetitions = HashMap::new();
        for pos in positions {
            *repetitions.entry(pos).or_insert(0) += 1;
        }

        for (_pos, repeat) in repetitions {
            if repeat >= repetition_limit.unwrap() {
                return true;
            }
        }
        return false;
    }

    fn develop_tree(&mut self) -> () {
        for _ in 0..self.simulations_per_move {
            /* Select a leaf node */
            let path_to_selection = self.select();

            let repetition_reached = self.detect_repetition(&path_to_selection);
            let leaf_id: NodeIndex = *path_to_selection.last().unwrap();
            let leaf_pos = &self.search_tree.node_weight(leaf_id).unwrap().position;

            let (eval, per_move_val);
            if leaf_pos.is_over() || repetition_reached {
                eval = if repetition_reached {
                    0.0
                } else {
                    GameColor::to_idx(leaf_pos.get_winner()) as f32
                };
            } else {
                /* Run value function once to obtain "simulation" value and initial children scores (probabilities) */
                (eval, per_move_val) = self.simulate(leaf_id);

                /* Expand leaf and assign initial scores */
                self.create_children(leaf_id, per_move_val);
            }

            /* back propagate the position score to the parents */
            self.backpropagate(&path_to_selection, eval)
        }
    }

    /* Return path to selected leaf node */
    fn select(&self) -> Vec<NodeIndex> {
        let mut path: Vec<NodeIndex> = vec![];

        let mut node_id = self.root.unwrap();
        loop {
            path.push(node_id);
            let node = self.search_tree.node_weight(node_id).unwrap();

            /* Node is leaf, done */
            if node.position.is_over() || self.search_tree.edges(node_id).next().is_none() {
                return path;
            }

            /* Node is not a leaf, choose best child and continue in it's sub tree */
            node_id = self
                .search_tree
                .edges(node_id)
                .map(|edge| edge.target())
                .max_by(|&child1_id, &child2_id| {
                    let child1 = self.search_tree.node_weight(child1_id).unwrap();
                    let child2 = self.search_tree.node_weight(child2_id).unwrap();
                    let val1 = self.calc_selection_heuristic(node, child1);
                    let val2 = self.calc_selection_heuristic(node, child2);
                    return val1.partial_cmp(&val2).unwrap();
                })
                .unwrap();
        }
    }

    fn calc_selection_heuristic(
        &self,
        parent: &MCTSNode<Game::Position>,
        child: &MCTSNode<Game::Position>,
    ) -> f32 {
        let exploit = if child.simulations_n == 0 {
            0.0
        } else {
            child.score_w / child.simulations_n as f32
        };

        let explore = self.exploration_param_c
            * child.init_score
            * ((parent.simulations_n as f32).sqrt() / (1 + child.simulations_n) as f32);

        return exploit + explore;
    }

    fn create_children(
        &mut self,
        parent_id: NodeIndex,
        per_move_init_score: Vec<(Game::Move, f32)>,
    ) {
        let parent_pos = self.search_tree.node_weight(parent_id).unwrap().position;
        assert!(!parent_pos.is_over());

        // TODO remove
        let moves_actual: HashSet<Game::Move> =
            HashSet::from_iter(per_move_init_score.iter().map(|(m, _p)| *m));
        let moves_expected: HashSet<Game::Move> =
            HashSet::from_iter(parent_pos.get_legal_moves().iter().map(|x| *x));
        assert!(moves_actual == moves_expected);

        for (m, p) in per_move_init_score {
            let leaf_pos = parent_pos.get_moved_position(m);
            let leaf_id = self
                .search_tree
                .add_node(MCTSNode::from_position(leaf_pos, p));
            self.search_tree.add_edge(parent_id, leaf_id, m);
        }
    }

    fn simulate(&mut self, leaf_id: NodeIndex) -> (f32, Vec<(Game::Move, f32)>) {
        let position = &self.search_tree.node_weight(leaf_id).unwrap().position;
        assert!(!position.is_over());
        return self.value_func.evaluate(&position);
    }

    fn backpropagate(&mut self, path: &Vec<NodeIndex>, score: f32) {
        for node_id in path {
            let mut node = self.search_tree.node_weight_mut(*node_id).unwrap();
            node.simulations_n += 1;
            node.score_w += match node.position.get_turn() {
                /* This is confusing: the score is 1 if player1 wins and -1 if player2 wins,
                 * nevertheless we apply -score if its player1 turn in the position.
                 * The reason is, the score of the nodes should represent the evaluation of the PARENT node towards
                 * the possible moves, leading to the children nodes.
                 * If its player1 turn in the position, the evaluation should be relative to player2 turn in the parent position.
                 *
                 * TODO: better is to save the score on the edges, more intuitive
                 */
                GameColor::Player1 => -score,
                GameColor::Player2 => score,
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
                if &self.search_tree.node_weight(node).unwrap().position == position {
                    return Some(node);
                }
                // Add children to next layer
                for edge in self.search_tree.edges(node).into_iter() {
                    next_layer.push(edge.target())
                }
            }
            layer = next_layer;
        }
        return None;
    }

    fn remove_all_but_subtree(&mut self, sub_tree_root: NodeIndex) {
        if self.root.unwrap() == sub_tree_root {
            return;
        }

        // In petgraph, when you remove a node, the indices of the other nodes
        // change. So instead of removing nodes from the current tree, we copy
        // the sub tree to a new graph
        let mut new_tree = DiGraph::new();
        let new_root = new_tree.add_node(*self.search_tree.node_weight(sub_tree_root).unwrap());
        let mut nodes = vec![(sub_tree_root, new_root)];

        while !nodes.is_empty() {
            let (parent_old, parent_new) = nodes.pop().unwrap();

            for edge in self.search_tree.edges(parent_old).into_iter() {
                let child_old = edge.target();
                let child_data = self.search_tree.node_weight(child_old).unwrap();
                let child_new = new_tree.add_node(*child_data);
                new_tree.add_edge(parent_new, child_new, *edge.weight());

                nodes.push((child_old, child_new));
            }
        }

        self.search_tree = new_tree;
        self.root = Some(new_root);
    }

    pub fn calc_moves_probabilities(
        &mut self,
        position: &Game::Position,
    ) -> Vec<(Game::Move, f32)> {
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
            let root = MCTSNode::from_position(*position, 1.0);
            self.root = Some(self.search_tree.add_node(root));
        }
        assert!(
            position
                == &self
                    .search_tree
                    .node_weight(self.root.unwrap())
                    .unwrap()
                    .position
        );

        // Run all simulations
        self.develop_tree();

        // create moves vector (move, sim_count)
        let moves_and_simcounts = self
            .search_tree
            .edges(self.root.unwrap())
            .into_iter()
            .map(|edge| {
                let child = self.search_tree.node_weight(edge.target()).unwrap();
                (edge.weight(), child.simulations_n)
            })
            .collect_vec();

        // normalize sim counts to create a valid distribution -> (move, simcount / simcount_total)
        let simcount_total: u32 = moves_and_simcounts
            .iter()
            .map(|&(_, simcount)| simcount)
            .sum();
        return moves_and_simcounts
            .into_iter()
            .map(|(m, simcount)| (*m, simcount as f32 / simcount_total as f32))
            .collect_vec();
    }

    pub fn choose_move_from_probabilities(
        &self,
        moves_probs: &Vec<(Game::Move, f32)>,
    ) -> Option<Game::Move> {
        if moves_probs.len() == 0 {
            return None;
        }

        if self.temperature == 0.0 {
            let (m, _p) = moves_probs
                .into_iter()
                .max_by(|(_m1, p1), (_m2, p2)| p1.total_cmp(p2))
                .unwrap();
            return Some(*m);
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
            let distribution = WeightedIndex::new(&probabilities).unwrap();
            return Some(moves_probs[distribution.sample(&mut rand::thread_rng())].0);
        }
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        assert!(temperature >= 0.0);
        self.temperature = temperature;
    }
}

impl<Game: IGame> GamePlayer<Game> for MCTSPlayer<Game> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(position);
        return self.choose_move_from_probabilities(&moves);
    }
}

pub trait ValueFunction<Game: IGame> {
    /// Evaluate a position
    ///
    /// position - The position to evaluate
    ///
    /// Returns a tuple of a scalar value score of the position and per-move scores/probabilities.
    /// The scalar is the current position value in range [-1,1]. 1 if player1 is winning and -1 if player2 is winning
    /// The per-move probabilities should have a sum of 1, greater value is a better move
    fn evaluate(&mut self, position: &Game::Position) -> (f32, Vec<(Game::Move, f32)>);
}

pub struct ValueFunctionRand {}
impl ValueFunctionRand {
    pub fn new() -> Self {
        Self {}
    }
}

impl<Game: IGame> ValueFunction<Game> for ValueFunctionRand {
    fn evaluate(&mut self, position: &Game::Position) -> (f32, Vec<(Game::Move, f32)>) {
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

        return (val, moves_probs);
    }
}
