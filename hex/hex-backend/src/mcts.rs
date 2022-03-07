use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::prelude::{IteratorRandom, SliceRandom};
use std::collections::HashMap;

use crate::hex_game::{self, Color};
use crate::simple_players::HexPlayerRand;
use hex_game::{HexGame, HexPlayer, HexPosition, Location};

/// Monte Carlo Tree Search (MCTS) implementation
/// TODO make this general and not Hex-specific

struct MCTSNode {
    position: HexPosition,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    /// Float because could be half for games with ties
    score_w: f32,
}

impl MCTSNode {
    pub fn from_position(pos: HexPosition) -> Self {
        Self {
            position: pos,
            simulations_n: 0,
            score_w: 0.0,
        }
    }

    pub fn get_expected_reward(&self) -> f32 {
        assert!(self.simulations_n > 0);
        return (self.score_w as f32) / (self.simulations_n as f32);
    }
}

pub struct MCTSPlayer {
    search_tree: DiGraph<MCTSNode, Location>,

    // exploration_parameter_c: f32, TODO
    simulations_per_move: u32,
}

impl MCTSPlayer {
    pub fn new() -> Self {
        MCTSPlayer::with_simulations_per_move(100)
    }
    pub fn with_simulations_per_move(simulations_per_move: u32) -> Self {
        Self {
            search_tree: DiGraph::new(),
            // exploration_parameter_c: (2 as f32).sqrt(),
            simulations_per_move: simulations_per_move,
        }
    }

    fn develop_tree(&mut self, root_id: NodeIndex<u32>, us: hex_game::Color) -> () {
        for _ in 1..self.simulations_per_move {
            let selection = self.select_node(root_id);
            let leaf_id = selection.0;
            let path = selection.1;
            let leaf = self.search_tree.node_weight(leaf_id).unwrap();
            let game_winner = self.simulate_playout(&leaf.position, us);
            self.backpropagate(path, game_winner);
        }
    }

    // Find an unexplored node and add it to the search tree, return (leaf_ID, [nodes on path]) or None if non found.
    // The list of nodes on path is used for back propagation.
    fn select_node(&mut self, root_id: NodeIndex<u32>) -> (NodeIndex<u32>, Vec<NodeIndex<u32>>) {
        let selection = self.select_node0(root_id);
        let leaf_parent = selection.0;
        let m_option = selection.1;
        let mut path = selection.2;
        path.reverse();

        match m_option {
            None => return (leaf_parent, path),
            Some(m) => {
                let parent = self.search_tree.node_weight(leaf_parent).unwrap();
                let leaf_pos = parent.position.get_moved_position(m);
                let leaf = MCTSNode::from_position(leaf_pos);
                let leaf_id = self.search_tree.add_node(leaf);
                self.search_tree.add_edge(leaf_parent, leaf_id, m);
                path.push(leaf_id);
                return (leaf_id, path);
            }
        }
    }

    // Find an unexplored (leaf) node and return (leaf_parent, move, [nodes on path]), or None if non found.
    // The list of nodes on path is used for back propagation (returned in reverse order, shouldn't matter).
    fn select_node0(
        &self,
        parent_id: NodeIndex<u32>,
    ) -> (NodeIndex<u32>, Option<Location>, Vec<NodeIndex<u32>>) {
        let parent = self.search_tree.node_weight(parent_id).unwrap();
        if parent.position.is_over() {
            // Node has no children that can be explored
            return (parent_id, None, vec![parent_id]);
        }

        // TODO: This is slow
        let mut existing_children: HashMap<Location, NodeIndex<u32>> = HashMap::new();
        for edge in self.search_tree.edges(parent_id) {
            let child_id = edge.target();
            let m = edge.weight();
            existing_children.insert(*m, child_id);
        }

        let legal_moves = parent.position.get_legal_moves();
        assert!(legal_moves.len() > 0);

        // TODO need to expand ALL unexplored nodes, currently not possible with number of simulations.
        // First expand unexplored nodes
        for m in &legal_moves {
            if !existing_children.contains_key(&m) {
                return (parent_id, Some(*m), vec![parent_id]);
            }
        }

        let mut m_best: Option<(NodeIndex<u32>, f32)> = None;
        for m in &legal_moves {
            assert!(parent.position.is_valid_move(*m));
            let child_id = *existing_children.get(m).unwrap();
            let child = self.search_tree.node_weight(child_id).unwrap();
            let val = self.calc_selection_heuristic(parent, child);
            match m_best {
                None => m_best = Some((child_id, val)),
                Some((_, val_best)) => {
                    if val > val_best {
                        m_best = Some((child_id, val));
                    }
                }
            }
        }
        let chosen_child_id = m_best.unwrap().0;
        let mut res = self.select_node0(chosen_child_id);
        res.2.push(parent_id);
        return res;
    }

    fn calc_selection_heuristic(&self, parent: &MCTSNode, child: &MCTSNode) -> f32 {
        let c = 2 as f32;
        let exploit = (child.score_w as f32) / (child.simulations_n as f32);
        let explore =
            c * ((parent.simulations_n as f32).ln() / (child.simulations_n as f32)).sqrt();
        return exploit + explore;
    }

    fn simulate_playout(&self, pos: &HexPosition, us: hex_game::Color) -> Option<Color> {
        // Play randomly and return the simulation game result
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();
        let mut sim_game = HexGame::from_position(pos, &mut player1, &mut player2);
        sim_game.play_until_over()
    }

    fn backpropagate(&mut self, path: Vec<NodeIndex<u32>>, winner: Option<Color>) {
        for node_id in path {
            let mut node = self.search_tree.node_weight_mut(node_id).unwrap();
            node.simulations_n += 1;
            node.score_w += match winner {
                None => 0.5,
                Some(color) => {
                    // Notice - this score is used by the parent, so the values represent value for parent.
                    if color == node.position.get_turn() {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
        }
    }

    fn get_best_child_move(&self, node_id: NodeIndex<u32>) -> Option<Location> {
        let edges = self.search_tree.edges(node_id);
        let edges_with_rewards: Vec<_> = edges
            .into_iter()
            .map(|edge| {
                let child = self.search_tree.node_weight(edge.target()).unwrap();
                (edge, child.get_expected_reward())
            })
            .collect();
        if edges_with_rewards.len() == 0 {
            return None;
        }
        let best_reward = edges_with_rewards
            .iter()
            .max_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap()
            .1;
        let edges_with_best_reward = edges_with_rewards.iter().filter_map(|&(edge, reward)| {
            if reward == best_reward {
                Some(edge)
            } else {
                None
            }
        });
        let chosen_edge = edges_with_best_reward.choose(&mut rand::thread_rng());
        return Some(chosen_edge.unwrap().weight().clone());
    }
}

impl HexPlayer for MCTSPlayer {
    fn next_move(&mut self, pos: &HexPosition) -> Option<Location> {
        // Init search tree with one root node
        assert!(self.search_tree.node_count() == 0);
        let root = MCTSNode::from_position(pos.clone());
        let root_id = self.search_tree.add_node(root);

        // Develop tree
        self.develop_tree(root_id, pos.get_turn());
        let m = self.get_best_child_move(root_id);

        self.search_tree.clear();

        return m;
    }
}
