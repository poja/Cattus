use crate::chess::chess_game::{ChessGame, ChessMove, ChessPosition};
use crate::game::common::{GamePlayer, GamePosition};
use crate::game::mcts::MCTSPlayer;
use crate::utils::Builder;
use itertools::Itertools;
use std::collections::HashMap;
// use std::fs::{File, OpenOptions};
use std::io;
// use std::io::prelude::*;
// use std::path::Path;

struct GoParams {
    searchmoves: Vec<String>,
    ponder: bool,
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movestogo: Option<u32>,
    depth: Option<u32>,
    nodes: Option<u32>,
    movetime: Option<u64>,
    infinite: bool,
}
impl GoParams {
    pub fn new() -> Self {
        Self {
            searchmoves: Vec::new(),
            ponder: false,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movestogo: None,
            depth: None,
            nodes: None,
            movetime: None,
            infinite: false,
        }
    }
}

struct Engine {
    player_builder: Box<dyn Builder<MCTSPlayer<ChessGame>>>,
    options: HashMap<String, String>,
    player: Option<MCTSPlayer<ChessGame>>,
    position: Option<ChessPosition>,
    best_move: Option<ChessMove>,
}

impl Engine {
    pub fn new(player_builder: Box<dyn Builder<MCTSPlayer<ChessGame>>>) -> Self {
        Self {
            player_builder,
            options: HashMap::new(),
            player: None,
            position: None,
            best_move: None,
        }
    }

    pub fn cmd_uci(&self) {
        self.send_response("id name _PROJECT_NAME_TODO_ v1.0.0");
        self.send_response("id author Barak Ugav Yishai Gronich");

        /* TODO send options */

        self.send_response("uciok");
    }

    pub fn cmd_isready(&self) {
        self.send_response("readyok");
    }

    pub fn cmd_setoption(&mut self, args: HashMap<String, String>) {
        self.options.extend(args);
    }

    pub fn cmd_ucinewgame(&mut self) {
        self.player = Some(self.player_builder.build());
    }

    pub fn cmd_position(&mut self, args: HashMap<String, String>) {
        if args.contains_key("fen") == args.contains_key("startpos") {
            panic!("position cmd requires either fen or startpos");
        }
        let fen = (args
            .get("fen")
            .unwrap_or(&"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()))
        .to_owned();

        let mut pos = ChessPosition::from_str(&fen);
        for move_str in args.get("moves").unwrap_or(&"".to_string()).split(' ') {
            if move_str.is_empty() {
                continue;
            }
            let m = ChessMove::from_lan(move_str)
                .unwrap_or_else(|_| panic!("Invalid move: {}", move_str));
            pos = pos.get_moved_position(m);
        }
        self.position = Some(pos);
    }

    pub fn cmd_go(&mut self, args: HashMap<String, String>) {
        let mut go_args = GoParams::new();
        go_args.searchmoves = args
            .get("searchmoves")
            .unwrap_or(&"".to_string())
            .split(' ')
            .map(|s| s.to_string())
            .collect_vec();
        go_args.ponder = args.contains_key("ponder");
        go_args.wtime = args.get("wtime").and_then(|s| s.parse::<u64>().ok());
        go_args.btime = args.get("btime").and_then(|s| s.parse::<u64>().ok());
        go_args.winc = args.get("winc").and_then(|s| s.parse::<u64>().ok());
        go_args.binc = args.get("binc").and_then(|s| s.parse::<u64>().ok());
        go_args.movestogo = args.get("movestogo").and_then(|s| s.parse::<u32>().ok());
        go_args.depth = args.get("depth").and_then(|s| s.parse::<u32>().ok());
        go_args.nodes = args.get("nodes").and_then(|s| s.parse::<u32>().ok());
        go_args.movetime = args.get("movetime").and_then(|s| s.parse::<u64>().ok());
        go_args.infinite = args.contains_key("infinite");

        let player = self.player.as_mut().unwrap();
        self.best_move = Some(player.next_move(&self.position.unwrap()).unwrap());
        self.send_response(format!("bestmove {}", self.best_move.unwrap()));
    }

    pub fn cmd_stop(&mut self) {}

    fn send_response<S: Into<String>>(&self, s: S) {
        let s = s.into();
        log(format!("[To GUI] {}", s));
        println!("{}", s);
    }
}

pub struct UCI {
    engine: Engine,
}

impl UCI {
    pub fn new(player_builder: Box<dyn Builder<MCTSPlayer<ChessGame>>>) -> Self {
        Self {
            engine: Engine::new(player_builder),
        }
    }

    pub fn run(&mut self) {
        let mut line = String::new();
        loop {
            line.clear();
            let read_res = io::stdin().read_line(&mut line);
            if read_res.is_err() {
                eprintln!("read line error: {}", read_res.err().unwrap());
                continue;
            }
            line = line.trim().to_string();
            log(format!("[From GUI] {}", line));
            let parsed = self.parse_command(line.clone());
            if parsed.is_none() {
                continue;
            }
            let (command, args) = parsed.unwrap();

            if command == "uci" {
                self.engine.cmd_uci();
            } else if command == "isready" {
                self.engine.cmd_isready();
            } else if command == "setoption" {
                self.engine.cmd_setoption(args);
            } else if command == "ucinewgame" {
                self.engine.cmd_ucinewgame();
            } else if command == "position" {
                self.engine.cmd_position(args);
            } else if command == "go" {
                self.engine.cmd_go(args);
            } else if command == "stop" {
                self.engine.cmd_stop();
            // } else if command == "ponderhit" {
            //     println!("uciok");
            // } else if command == "start" {
            //     println!("uciok");
            // } else if command == "fen" {
            //     println!("uciok");
            // } else if command == "xyzzy" {
            //     println!("uciok");
            } else if command == "quit" {
                return;
            } else {
                eprintln!("unknown command {command}");
            }
        }
    }

    fn parse_command(&self, s: String) -> Option<(String, HashMap<String, String>)> {
        let commands_format = [
            ("uci", vec![]),
            ("isready", vec![]),
            ("setoption", vec!["name", "value"]),
            ("ucinewgame", vec![]),
            ("position", vec!["fen", "startpos", "moves"]),
            (
                "go",
                vec![
                    "searchmoves",
                    "ponder",
                    "wtime",
                    "btime",
                    "winc",
                    "binc",
                    "movestogo",
                    "depth",
                    "nodes",
                    "mate",
                    "movetime",
                    "infinite",
                ],
            ),
            ("stop", vec![]),
            // ("ponderhit", vec![]),
            ("quit", vec![]),
        ];

        let command;
        let suffix;
        match s.chars().position(|c| c == ' ') {
            None => {
                command = &s[..];
                suffix = "";
            }
            Some(first_space) => {
                command = &s[..first_space];
                suffix = &s[first_space..];
            }
        };

        let command_format = commands_format.into_iter().find(|(c, _fmt)| command == *c);
        if command_format.is_none() {
            panic!("unknown command {command}");
        }
        let command_format = command_format.unwrap().1;

        let mut args = HashMap::new();
        let mut last_arg = None;
        for word in suffix.split(' ') {
            if word.is_empty() {
                continue;
            }
            if command_format.contains(&word) {
                if args.contains_key(word) {
                    panic!("double arg {word}");
                }
                args.insert(word.to_string(), "".to_string());
                last_arg = Some(word);
            } else {
                let val = args
                    .get_mut(last_arg.unwrap_or_else(|| panic!("unexpected argument '{}'", word)))
                    .unwrap();
                val.push(' ');
                val.push_str(word);
            }
        }

        args = args
            .into_iter()
            .map(|(key, val)| (key.trim().to_string(), val.trim().to_string()))
            .collect();
        Some((command.to_string(), args))
    }
}

fn log<S: Into<String>>(_s: S) {
    // let s = s.into();
    // let path = Path::new("C:\\code\\rl\\log.txt");
    // if !path.exists() {
    //     File::create(path).expect("failed to create log file");
    // }
    // let mut file = OpenOptions::new()
    //     .write(true)
    //     .append(true)
    //     .open(path)
    //     .unwrap();
    // writeln!(file, "{}", s).expect("failed to append to log file");
}
