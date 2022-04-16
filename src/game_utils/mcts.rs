use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rand::prelude::IteratorRandom;

use crate::game_utils::game::{GameColor, GamePlayer, GamePosition, IGame};
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
    pub fn from_position(pos: &Position) -> Self {
        Self {
            position: *pos,
            simulations_n: 0,
            score_w: 0.0,
        }
    }

    pub fn get_expected_reward(&self) -> f32 {
        assert!(self.simulations_n > 0);
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
            self.select_and_simulate(root_id);
        }
    }

    // Select a node to run simulation from (and add it to the search tree), return (leaf_ID, [nodes on path]).
    // The list of nodes on path is used for back propagation.
    fn select_and_simulate(&mut self, root_id: NodeIndex<u32>) {
        let mut path: Vec<NodeIndex<u32>> = vec![];

        let mut node_id = root_id;
        loop {
            path.push(node_id);
            let node = self.search_tree.node_weight(node_id).unwrap();

            /* Node has no children, perform a (very) short simulation and done */
            if node.position.is_over() {
                let (player1_wins, player2_wins, draws) = match node.position.get_winner() {
                    Some(color) => match color {
                        GameColor::Player1 => (1, 0, 0),
                        GameColor::Player2 => (0, 1, 0),
                    },
                    None => (0, 0, 1),
                };
                self.backpropagate(&path, player1_wins, player2_wins, draws);
                return;
            }

            // Node has no children in tree, expand all and simulate one simulation per child
            if self.search_tree.edges(node_id).next().is_none() {
                let parent_pos = node.position;

                let (mut player1_wins, mut player2_wins, mut draws) = (0, 0, 0);
                for m in parent_pos.get_legal_moves() {
                    let leaf_pos = parent_pos.get_moved_position(m);
                    let leaf_id = self
                        .search_tree
                        .add_node(MCTSNode::from_position(&leaf_pos));
                    self.search_tree.add_edge(node_id, leaf_id, m);
                    let mut leaf = self.search_tree.node_weight_mut(leaf_id).unwrap();
                    match MCTSPlayer::<Game>::simulate_playout(&leaf_pos) {
                        Some(color) => {
                            if parent_pos.get_turn() == color {
                                leaf.score_w += 1.0;
                            }
                            match color {
                                GameColor::Player1 => player1_wins += 1,
                                GameColor::Player2 => player2_wins += 1,
                            };
                        }
                        None => {
                            leaf.score_w += 0.5;
                            draws += 1;
                        }
                    }
                    leaf.simulations_n += 1;
                }

                /* back propagate a single time for all children */
                self.backpropagate(&path, player1_wins, player2_wins, draws);
                return;
            }

            /* Node children already been expanded, choose best one and continue in it's sub tree */
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
        let exploit = (child.score_w as f32) / (child.simulations_n as f32);
        let explore = self.exploration_param_c
            * ((parent.simulations_n as f32).ln() / (child.simulations_n as f32)).sqrt();
        return exploit + explore;
    }

    fn simulate_playout(pos: &Game::Position) -> Option<GameColor> {
        // Play randomly and return the simulation game result
        let mut player1 = PlayerRand::new();
        let mut player2 = PlayerRand::new();
        Game::play_until_over(pos, &mut player1, &mut player2).1
    }

    fn backpropagate(
        &mut self,
        path: &Vec<NodeIndex<u32>>,
        player1_wins: u32,
        player2_wins: u32,
        draws: u32,
    ) {
        let first_node = self.search_tree.node_weight(path[0]).unwrap();
        let games_num = player1_wins + player2_wins + draws;
        let mut applied_score = (draws as f32) / 2.0
            + match first_node.position.get_turn() {
                GameColor::Player1 => player2_wins,
                GameColor::Player2 => player1_wins,
            } as f32;

        for node_id in path {
            let mut node = self.search_tree.node_weight_mut(*node_id).unwrap();
            node.simulations_n += games_num;
            node.score_w += applied_score as f32;
            applied_score = games_num as f32 - applied_score;
        }
    }

    pub fn calc_moves_probabilities(
        &mut self,
        position: &Game::Position,
    ) -> Vec<(Game::Move, f32)> {
        // Init search tree with one root node
        assert!(self.search_tree.node_count() == 0);
        let root = MCTSNode::from_position(position);
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
        self.search_tree.clear();
        return self.choose_move_from_probabilities(&moves);
    }
}
