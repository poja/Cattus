use crate::game_utils::game::{GameColor, GamePlayer, GamePosition, IGame};
use crate::hex::hex_game::{HexGame, HexPosition, Hexagon, BOARD_SIZE};
use std::io::{BufRead, BufReader, Write};
use std::string::String;
use std::{io, process, thread, time};

/**
 * UXI (Universal Hex Interface), similar to UCI (Universal Chess Interface) is a general interface for
 * Hex engine communication. All communication is done by standard input and output, each command has '\n' at the end.
 *
 * Input commands (from host to engine):
 *      next_move [pos] [color]
 *          calculate the next move of the engine.
 *              [pos] - current position, string of 121 characters, e/r/b (empty, red, blue),
 *                  i'th character corresponding to the [i/11][i%11] tile
 *              [color] - the engine color, one character, r/b (red, blue)
 *      quit
 *          quit from the program, the engine should exit in 0.1 sec
 * Output commands (from engine to host):
 *      ready
 *          after engine was started, this is the first command it should issue to let the host know its ready
 *      move [indices]
 *          the next move of the engine
 *              indices - the engine move, two numbers with comma separation, "c,r"
 */

pub struct HexPlayerUXI {
    exe_filename: String,
    // err_filename: String,
    // err_file: Option<File>,
    process: Option<process::Child>,
}

impl HexPlayerUXI {
    pub fn new(exe_filename: &std::path::Path /*, err_filename: &std::path::Path*/) -> Self {
        Self {
            exe_filename: String::from(exe_filename.to_str().unwrap()),
            // err_filename: String::from(err_filename.to_str().unwrap()),
            // err_file: None,
            process: None,
        }
    }

    pub fn start(&mut self, engine_params: &Vec<String>) -> bool {
        if self.process.is_some() {
            println!("[UXIPlayer] Process is already launched");
            return false;
        }

        println!(
            "[UXIPlayer] launching engine: {} {:?}",
            self.exe_filename, engine_params
        );

        // assert!(self.err_file.is_none());
        // match File::create(self.err_filename) {
        //     Err(error) => {
        //         eprintln!("Failed to open error file for process: {}", error);
        //         self.err_file = None;
        //         return false;
        //     }
        //     Ok(file) => self.err_file = file,
        // }

        self.process = match process::Command::new(self.exe_filename.clone())
            .args(engine_params)
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            // .stderr(process::Stdio::from_raw_fd(
            //     self.err_file.unwrap().into_raw_fd(),
            // ))
            .spawn()
        {
            Err(error) => {
                eprintln!("[UXIPlayer] Failed to launch process: {}", error);
                None
            }
            Ok(process) => Some(process),
        };
        if self.process.is_none() {
            return false;
        }

        let r = self.receive_command();
        if r.is_none() {
            return false;
        }
        let resp = r.unwrap();
        let response: Vec<_> = resp.split(" ").collect();
        if response.is_empty() {
            return false;
        }
        match response[0] {
            "ready" => return true,
            _ => {
                eprintln!("[UXIPlayer] Unexpected command: {:?} (expected ready)", response);
                return false;
            }
        };
    }

    pub fn stop(&mut self) {
        if self.process.is_some() {
            /* be nice (for 0.1 sec) */
            self.send_command(String::from("quit"));
            thread::sleep(time::Duration::from_millis(100));

            let mut kill_needed = false;
            match self.process.as_mut().unwrap().try_wait() {
                Err(error) => {
                    eprintln!("[UXIPlayer] Failed to get engine process status: {}", error);
                    kill_needed = true;
                }
                Ok(status) => match status {
                    None => {
                        eprintln!("[UXIPlayer] Engine process ignored 'quit' command");
                        kill_needed = true;
                    }
                    Some(_) => { /* engine quit by it's own */ }
                },
            };
            if kill_needed {
                /* don't be nice */
                match self.process.as_mut().unwrap().kill() {
                    Err(error) => eprintln!("[UXIPlayer] Failed to kill process: {}", error),
                    Ok(_) => {}
                }
            }
            self.process = None;
        }
    }

    fn send_command(&mut self, cmd: String) {
        if self.process.is_none() {
            eprintln!("[UXIPlayer] Engine was not started.");
            return;
        }
        let process = self.process.as_mut().unwrap();
        let engine_stdin = process.stdin.as_mut().unwrap();
        match engine_stdin.write((String::from(cmd.trim()) + "\n").as_bytes()) {
            Err(error) => eprintln!("[UXIPlayer] Failed to pass command: {}", error),
            Ok(_) => {}
        }
        drop(engine_stdin);
    }

    fn receive_command(&mut self) -> Option<String> {
        if self.process.is_none() {
            eprintln!("[UXIPlayer] Engine was not started.");
            return None;
        }
        let process = self.process.as_mut().unwrap();
        let mut engine_stdout = BufReader::new(process.stdout.as_mut().unwrap());
        let mut output_line = String::new();

        match engine_stdout.read_line(&mut output_line) {
            Err(error) => {
                eprintln!("[UXIPlayer] Failed to read output from engine: {}", error);
                return None;
            }
            Ok(_) => {
                return Some(String::from(output_line.trim()));
            }
        }
    }
}

impl GamePlayer<HexGame> for HexPlayerUXI {
    fn next_move(
        &mut self,
        position: &<HexGame as IGame>::Position,
    ) -> Option<<HexGame as IGame>::Move> {
        let mut command = String::with_capacity(10 + BOARD_SIZE * BOARD_SIZE + 3);
        command.push_str("next_move ");
        position_to_uxi(position, &mut command);
        self.send_command(command);
        let r = self.receive_command();
        if r.is_none() {
            return None;
        }
        let resp = r.unwrap();
        let response: Vec<_> = resp.split(" ").collect();
        if response.is_empty() {
            return None;
        }
        match response[0] {
            "move" => {
                if response.len() != 2 {
                    eprintln!("[UXIPlayer] Expected \"move r,c\" format: \"{}\"", resp);
                    return None;
                }
                let m_str: Vec<_> = response[1].split(",").collect();
                if m_str.len() != 2 {
                    eprintln!("[UXIPlayer] Expected \"move r,c\" format: \"{}\"", resp);
                    return None;
                }
                let r = match m_str[0].parse::<usize>() {
                    Err(error) => {
                        eprintln!("[UXIPlayer] Failed to parse row index: {}", error);
                        return None;
                    }
                    Ok(row) => row,
                };
                let c = match m_str[1].parse::<usize>() {
                    Err(error) => {
                        eprintln!("[UXIPlayer] Failed to parse column index: {}", error);
                        return None;
                    }
                    Ok(column) => column,
                };
                return Some((r, c));
            }
            unknown_cmd => {
                eprintln!("[UXIPlayer] Unknown command: {}", unknown_cmd);
                return None;
            }
        };
    }
}

pub struct UXIEngine<'a> {
    player: &'a mut dyn GamePlayer<HexGame>,
}

impl<'a> UXIEngine<'a> {
    pub fn new(player: &'a mut dyn GamePlayer<HexGame>) -> Self {
        Self { player: player }
    }

    pub fn run(&mut self) {
        println!("ready");
        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("[UXIEngine] failed to read input");
            let args: Vec<_> = line.split_whitespace().collect();

            if args.is_empty() {
                continue;
            }
            match args[0] {
                "next_move" => {
                    if args.len() != 3 {
                        eprintln!("[UXIEngine] Expected position and color for next_move command.");
                        continue;
                    }
                    let pos_str = args[1];
                    let color_str = args[2];
                    match uxi_to_position(pos_str, color_str) {
                        None => {
                            eprintln!("[UXIEngine] Failed to parse position.");
                            continue;
                        }
                        Some(pos) => {
                            match self.player.next_move(&pos) {
                                None => println!("error"),
                                Some(m) => println!("move {},{}", m.0, m.1),
                            };
                        }
                    }
                }
                "quit" => {
                    break;
                }
                unknown_cmd => {
                    eprintln!("[UXIEngine] Unknown command: {}", unknown_cmd);
                }
            }
        }
    }
}

fn position_to_uxi(position: &HexPosition, s: &mut String) {
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            s.push(match position.get_tile(r, c) {
                Hexagon::Empty => 'e',
                Hexagon::Full(color) => match color {
                    GameColor::Player1 => 'r',
                    GameColor::Player2 => 'b',
                },
            });
        }
    }
    s.push(' ');
    s.push(match position.get_turn() {
        GameColor::Player1 => 'r',
        GameColor::Player2 => 'b',
    });
}

fn uxi_to_position(pos_str: &str, color_str: &str) -> Option<HexPosition> {
    let mut board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE] = [[Hexagon::Empty; BOARD_SIZE]; BOARD_SIZE];
    let mut i = 0;
    for tile in pos_str.chars() {
        if i >= BOARD_SIZE * BOARD_SIZE {
            eprintln!("Too many chars in position string");
            return None;
        }
        board[i / BOARD_SIZE][i % BOARD_SIZE] = match tile {
            'e' => Hexagon::Empty,
            'r' => Hexagon::Full(GameColor::Player1),
            'b' => Hexagon::Full(GameColor::Player2),
            unknown_tile => {
                eprintln!("Unknown tile: {}", unknown_tile);
                return None;
            }
        };
        i += 1;
    }
    if i != BOARD_SIZE * BOARD_SIZE {
        eprintln!("Too few chars in position string");
        return None;
    }
    let player = match color_str {
        "r" => GameColor::Player1,
        "b" => GameColor::Player2,
        unknown_player => {
            eprintln!("Unknown player: {}", unknown_player);
            return None;
        }
    };
    return Some(HexPosition::new_from_board(board, player));
}
