use itertools::Itertools;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashSet;
use std::iter::FromIterator;

use crate::game::common::{GameColor, GamePlayer, GamePosition, IGame, PlayerRand};

/// Monte Carlo Tree Search (MCTS) implementation

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

    exploration_param_c: f32,
    simulations_per_move: u32,
    value_func: Box<dyn ValueFunction<Game>>
}

impl<Game: IGame> MCTSPlayer<Game> {
        pub fn new(value_func: Box<dyn ValueFunction<Game>>) -> Self {
        MCTSPlayer::new_custom(100, (2 as f32).sqrt(), value_func)
    }
    pub fn new_custom(
        simulations_per_move: u32,
        exploration_param_c: f32,
        value_func: Box<dyn ValueFunction<Game>>,
    ) -> Self {
        Self {
            search_tree: DiGraph::new(),
            exploration_param_c: exploration_param_c,
            simulations_per_move: simulations_per_move,
            value_func: value_func,
        }
    }

    fn develop_tree(&mut self, root_id: NodeIndex<u32>) -> () {
        for _ in 0..self.simulations_per_move {
            /* Select a leaf node */
            let path_to_selection = self.select(root_id);

            /* Run value function once to obtain "simulation" value and initial children scores (probabilities) */
            let leaf_id: NodeIndex = *path_to_selection.last().unwrap();
            let (eval, per_move_val) = self.simulate(leaf_id);

            /* Expand leaf and assign initial scores */
            self.create_children(leaf_id, per_move_val);

            /* back propagate the position score to the parents */
            self.backpropagate(&path_to_selection, eval)
        }
    }

    /* Return path to selected leaf node */
    fn select(&self, root_id: NodeIndex<u32>) -> Vec<NodeIndex<u32>> {
        let mut path: Vec<NodeIndex<u32>> = vec![];

        let mut node_id = root_id;
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
        parent_id: NodeIndex<u32>,
        per_move_init_score: Vec<(Game::Move, f32)>,
    ) {
        let parent = self.search_tree.node_weight(parent_id).unwrap();
        if parent.position.is_over() {
            return;
        }
        let parent_pos = parent.position;

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
        if position.is_over() {
            let eval = match position.get_winner() {
                Some(w) => match w {
                    GameColor::Player1 => 1.0,
                    GameColor::Player2 => -1.0,
                },
                None => 0.0,
            };
            return (eval, vec![]);
        }
        return self.value_func.evaluate(&position);
    }

    fn backpropagate(&mut self, path: &Vec<NodeIndex<u32>>, score: f32) {
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

    pub fn calc_moves_probabilities(
        &mut self,
        position: &Game::Position,
    ) -> Vec<(Game::Move, f32)> {
        // Init search tree with one root node
        assert!(self.search_tree.node_count() == 0);
        let root = MCTSNode::from_position(*position, 1.0);
        let root_id = self.search_tree.add_node(root);

        // Develop tree
        self.develop_tree(root_id);

        // create moves vector (move, sim_count)
        let moves_and_simcounts = self
            .search_tree
            .edges(root_id)
            .into_iter()
            .map(|edge| {
                let child = self.search_tree.node_weight(edge.target()).unwrap();
                (edge.weight(), child.simulations_n)
            })
            .collect_vec();

        // normalize sim counts to create a valid distribution -> (move, simcount / simcount_total)
        let simcount_total: u64 = moves_and_simcounts
            .iter()
            .map(|&(_, simcount)| simcount as u64)
            .sum();
        return moves_and_simcounts
            .iter()
            .map(|&(m, simcount)| (*m, (simcount as f64 / simcount_total as f64) as f32))
            .collect_vec();
    }

    pub fn clear(&mut self) {
        self.search_tree.clear();
    }

    pub fn choose_move_from_probabilities(
        &self,
        moves_probs: &Vec<(Game::Move, f32)>,
    ) -> Option<Game::Move> {
        if moves_probs.len() == 0 {
            return None;
        }

        /* prob -> prob^temperature */
        let temperature = 1.0; /* TODO adjustable param */
        let probabilities = moves_probs
            .iter()
            .map(|(_m, p)| (*p as f64).powf(1.0 / temperature))
            .collect_vec();

        /* normalize, prob -> prob / (probs sum) */
        let probs_sum: f64 = probabilities.iter().sum();
        let probabilities = probabilities.iter().map(|p| p / probs_sum).collect_vec();
        let distribution = WeightedIndex::new(&probabilities).unwrap();
        return Some(moves_probs[distribution.sample(&mut rand::thread_rng())].0);
    }
}

impl<Game: IGame> GamePlayer<Game> for MCTSPlayer<Game> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(position);
        self.clear();
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
