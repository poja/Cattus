use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use rand::prelude::SliceRandom;

use crate::hex_game;
use hex_game::{HexGame, HexPlayer, HexPlayerRand, HexPosition, Location};

/// Monte Carlo Tree Search (MCTS) implementation
/// TODO make this general and not Hex-specific

struct MCTSNode {
    position: HexPosition,

    /// This is the variable n from UCT formula
    simulations_n: u32,

    /// This is the variable w from UCT formula
    score_w: i32,
}

impl MCTSNode {
    pub fn from_position(pos: HexPosition) -> Self {
        Self {
            position: pos,
            simulations_n: 0,
            score_w: 0,
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
        for i in 1..self.simulations_per_move {
            println!("Simulating: ({}/{})", i, self.simulations_per_move);
            match self.select_node(root_id) {
                Some((leaf_id, path)) => {
                    let leaf = self.search_tree.node_weight(leaf_id).unwrap();
                    let game_outcome = self.simulate_playout(&leaf.position, us);
                    self.backpropagate(path, game_outcome);
                }
                None => {
                    println!("No more unexplored nodes found, terminating simulation...");
                    break;
                }
            }
        }
    }

    // Find an unexplored node and add it to the search tree, return (leaf_ID, [nodes on path]) or None if non found.
    // The list of nodes on path is used for back propagation.
    fn select_node(
        &mut self,
        root_id: NodeIndex<u32>,
    ) -> Option<(NodeIndex<u32>, Vec<NodeIndex<u32>>)> {
        match self.select_node0(root_id) {
            // No more unexplored nodes
            None => None,
            // Found unexplored node, create and add to search tree
            Some((leaf_parent, m, mut path)) => {
                let parent = self.search_tree.node_weight(leaf_parent).unwrap();
                let leaf_pos = parent.position.get_moved_position(m);
                let leaf = MCTSNode::from_position(leaf_pos);

                let leaf_id = self.search_tree.add_node(leaf);
                self.search_tree.add_edge(leaf_parent, leaf_id, m);

                path.reverse();
                path.push(leaf_id);

                return Some((leaf_id, path));
            }
        }
    }

    // Find an unexplored (leaf) node and return (leaf_parent, move, [nodes on path]), or None if non found.
    // The list of nodes on path is used for back propagation (returned in reverse order, shouldn't matter).
    fn select_node0(
        &self,
        parent_id: NodeIndex<u32>,
    ) -> Option<(NodeIndex<u32>, Location, Vec<NodeIndex<u32>>)> {
        let parent = self.search_tree.node_weight(parent_id).unwrap();
        let win_status = parent.position.get_winner();
        if win_status.0 == true {
            // Node has no children that can be explored
            return None;
        }

        // TODO: This is slow
        let mut existing_children: HashMap<Location, NodeIndex<u32>> = HashMap::new();
        for edge in self.search_tree.edges(parent_id) {
            let child_id = edge.target();
            let m = edge.weight();
            existing_children.insert(*m, child_id);
        }

        let mut legal_moves = parent.position.get_legal_moves();
        assert!(legal_moves.len() > 0);
        legal_moves.shuffle(&mut rand::thread_rng());

        // Select successive child nodes randomly until a leaf node is reached
        for m in legal_moves {
            assert!(parent.position.is_valid_move(m));

            match existing_children.get(&m) {
                // Explored position, child already exist in tree, continue exploring his sub tree
                Some(existing_child_id) => {
                    match self.select_node0(*existing_child_id) {
                        // Child sub tree has no unexplored leafs, continue to next child
                        None => continue,
                        // Found unexplored position in child sub tree
                        Some((leaf_parent, m, mut path)) => {
                            path.push(parent_id);
                            return Some((leaf_parent, m, path));
                        }
                    }
                }
                // Unexplored position
                None => {
                    return Some((parent_id, m, vec![parent_id]));
                }
            }
        }
        return None;
    }

    /// Returns 1 if "we" won, -1 if lost, 0 if draw
    fn simulate_playout(&self, pos: &HexPosition, us: hex_game::Color) -> i32 {
        // Play randomly and return the simulation game result
        let mut player1 = HexPlayerRand::new();
        let mut player2 = HexPlayerRand::new();
        let mut sim_game = HexGame::from_position(pos, &mut player1, &mut player2);
        match sim_game.play_until_over() {
            Some(winner) => {
                if winner == us {
                    1
                } else {
                    -1
                }
            }
            None => 0,
        }
    }

    fn backpropagate(&mut self, path: Vec<NodeIndex<u32>>, game_outcome: i32) {
        for node_id in path {
            let mut node = self.search_tree.node_weight_mut(node_id).unwrap();
            node.simulations_n += 1;
            node.score_w += game_outcome;
        }
    }

    fn get_best_child_move(&self, node_id: NodeIndex<u32>) -> Option<Location> {
        let mut best: Option<(Location, f32)> = None;
        for edge in self.search_tree.edges(node_id) {
            let child_id = edge.target();
            let m = edge.weight();
            let child = self.search_tree.node_weight(child_id).unwrap();
            let score = child.get_expected_reward();

            match best {
                None => best = Some((*m, score)),
                Some((_, best_score)) => {
                    if score > best_score {
                        best = Some((*m, score));
                    }
                }
            }
        }
        match best {
            Some((m, _)) => Some(m),
            None => None,
        }
    }
}

impl HexPlayer for MCTSPlayer {
    fn next_move(&mut self, pos: &HexPosition) -> Location {
        // Init search tree with one root node
        assert!(self.search_tree.node_count() == 0);
        let root = MCTSNode::from_position(pos.clone());
        let root_id = self.search_tree.add_node(root);

        // Develop tree
        self.develop_tree(root_id, pos.get_turn());
        let m = self.get_best_child_move(root_id);

        self.search_tree.clear();

        return m.unwrap();
    }
}
