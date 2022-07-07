use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::prelude::IteratorRandom;

use crate::game_utils::game::{GamePlayer, GamePosition, IGame};
use crate::hex::simple_players::PlayerRand;

/// Monte Carlo Tree Search (MCTS) implementation

struct MCTSNode<Position: GamePosition> {
    position: Position,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    /// Float because could be half for games with ties
    score_w: f32,
}

impl<Position: GamePosition> MCTSNode<Position> {
    pub fn from_position(pos: Position) -> Self {
        Self {
            position: pos,
            simulations_n: 0,
            score_w: 0.0,
        }
    }

    pub fn get_expected_reward(&self) -> f32 {
        if self.simulations_n == 0 {
            return self.score_w;
        }
        return (self.score_w as f32) / (self.simulations_n as f32);
    }
}

pub struct MCTSPlayer<Game: IGame> {
    search_tree: DiGraph<MCTSNode<Game::Position>, Game::Move>,

    exploration_param_c: f32,
    simulations_per_move: u32,
}

impl<Game: IGame> MCTSPlayer<Game> {
    pub fn new() -> Self {
        MCTSPlayer::new_custom(100, (2 as f32).sqrt())
    }
    pub fn new_custom(simulations_per_move: u32, exploration_param_c: f32) -> Self {
        Self {
            search_tree: DiGraph::new(),
            exploration_param_c: exploration_param_c,
            simulations_per_move: simulations_per_move,
        }
    }

    fn develop_tree(&mut self, root_id: NodeIndex<u32>) -> () {
        for _ in 1..self.simulations_per_move {
            /* Select a leaf node */
            let path_to_selection = self.select(root_id);

            /* Expand leaf */
            let leaf_id: NodeIndex = *path_to_selection.last().unwrap();
            self.create_children(leaf_id);

            /* Simulate a single time from leaf */
            let score = self.simulate(&path_to_selection);

            /* back propagate the score to the parents */
            self.backpropagate(&path_to_selection, score)
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
        if child.simulations_n == 0 {
            return f32::MAX;
        }
        let exploit = (child.score_w as f32) / (child.simulations_n as f32);
        let explore = self.exploration_param_c
            * ((parent.simulations_n as f32).ln() / (child.simulations_n as f32)).sqrt();
        return exploit + explore;
    }

    fn create_children(&mut self, parent_id: NodeIndex<u32>) {
        let parent = self.search_tree.node_weight(parent_id).unwrap();
        if parent.position.is_over() {
            return;
        }
        let parent_pos = parent.position;
        for m in parent.position.get_legal_moves() {
            let leaf_pos = parent_pos.get_moved_position(m);
            let leaf_id = self
                .search_tree
                .add_node(MCTSNode::from_position(leaf_pos));
            self.search_tree.add_edge(parent_id, leaf_id, m);
        }
    }

    fn simulate(&self, path_to_selection: &Vec<NodeIndex<u32>>) -> f32 {
        let node_id: NodeIndex = *path_to_selection.last().unwrap();
        let node = self.search_tree.node_weight(node_id).unwrap();

        let score = MCTSPlayer::<Game>::simulate_playout(&node.position);

        let root_id: NodeIndex = *path_to_selection.last().unwrap();
        let root = self.search_tree.node_weight(root_id).unwrap();
        if root.position.get_turn() == node.position.get_turn() {
            return score;
        } else {
            return 1.0 - score;
        }
    }

    fn simulate_playout(pos: &Game::Position) -> f32 {
        let winner;
        if pos.is_over() {
            winner = pos.get_winner();
        } else {
            // Play randomly and return the simulation game result
            let mut player1 = PlayerRand::new();
            let mut player2 = PlayerRand::new();
            winner = Game::play_until_over(pos, &mut player1, &mut player2).1
        }
        return match winner {
            Some(color) => {
                if color == pos.get_turn() {
                    1.0
                } else {
                    0.0
                }
            }
            None => 0.5,
        };
    }

    fn backpropagate(&mut self, path: &Vec<NodeIndex<u32>>, score: f32) {
        let mut applied_score = score;
        for node_id in path {
            let mut node = self.search_tree.node_weight_mut(*node_id).unwrap();
            node.simulations_n += 1;
            node.score_w += applied_score as f32;
            applied_score = 1.0 - applied_score;
        }
    }

    pub fn calc_moves_probabilities(
        &mut self,
        position: &Game::Position,
    ) -> Vec<(Game::Move, f32)> {
        // Init search tree with one root node
        assert!(self.search_tree.node_count() == 0);
        let root = MCTSNode::from_position(*position);
        let root_id = self.search_tree.add_node(root);

        // Develop tree
        self.develop_tree(root_id);

        let moves_w_exp_reward = self
            .search_tree
            .edges(root_id)
            .into_iter()
            .map(|edge| {
                let child = self.search_tree.node_weight(edge.target()).unwrap();
                (edge.weight(), child.get_expected_reward().exp())
            })
            .collect::<Vec<_>>();

        let exp_sum: f32 = moves_w_exp_reward.iter().map(|&(_, exp_r)| exp_r).sum();

        let moves_w_probs = moves_w_exp_reward
            .iter()
            .map(|&(m, exp_r)| (*m, exp_r / exp_sum));
        return moves_w_probs.collect::<Vec<_>>();
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

        let highest_prob = moves_probs
            .iter()
            .max_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap()
            .1;
        let moves_w_highest_prob =
            moves_probs.iter().filter_map(
                |&(m, prob)| {
                    if prob >= highest_prob {
                        Some(m)
                    } else {
                        None
                    }
                },
            );
        return moves_w_highest_prob.choose(&mut rand::thread_rng());
    }
}

impl<Game: IGame> GamePlayer<Game> for MCTSPlayer<Game> {
    fn next_move(&mut self, position: &Game::Position) -> Option<Game::Move> {
        let moves = self.calc_moves_probabilities(position);
        self.clear();
        return self.choose_move_from_probabilities(&moves);
    }
}
