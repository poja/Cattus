use crate::hex_game::{Color, HexGame, HexPlayer, HexPosition, Hexagon, Location, BOARD_SIZE};
use rand::Rng;
use std::io::{BufRead, BufReader, Write};
use std::string::String;
use std::{io, process, thread, time};

/**
 * UXI (Universal Hex Interface), similar to UCI (Universal Chess Interface) is general interface for
 * Hex engine communication. All communication is done by standard input and output, each command has '\n' at the end.
 *
 * Input commands (from host to engine):
 *      next_move [pos] [color]
 *          pos - current position, string of 121 characters, e/r/b (empty, red, blue),
 *              i'th character corresponding to the [i/11][i%11] tile]
 *          color - the engine color, one character, r,b (red, blue)
 *      quit
 * Output commands (from engine to host):
 *      move [indices]
 *          indices - the engine move, two numbers with comma separation, "c,r"
 */

struct HexPlayerUXI {
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

    pub fn start(&mut self) -> bool {
        if self.process.is_some() {
            println!("Process is already launched");
            return false;
        }

        // assert!(self.err_file.is_none());
        // match File::create(self.err_filename) {
        //     Err(_) => {
        //         eprintln!("Failed to open error file for process");
        //         self.err_file = None;
        //         return false;
        //     }
        //     Ok(file) => self.err_file = file,
        // }

        self.process = match process::Command::new(self.exe_filename.clone())
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            // .stderr(process::Stdio::from_raw_fd(
            //     self.err_file.unwrap().into_raw_fd(),
            // ))
            .spawn()
        {
            Err(_) => {
                eprintln!("Failed to launch process");
                None
            }
            Ok(process) => Some(process),
        };
        return self.process.is_some();
    }

    pub fn stop(&mut self) {
        if self.process.is_some() {
            /* be nice (for 0.1 sec) */
            self.send_command(String::from("quit"));
            thread::sleep(time::Duration::from_millis(100));

            let mut kill_needed = false;
            match self.process.as_mut().unwrap().try_wait() {
                Err(_) => {
                    eprintln!("Failed to get engine process status");
                    kill_needed = true;
                }
                Ok(status) => match status {
                    None => {
                        eprintln!("Engine process ignored 'quit' command");
                        kill_needed = true;
                    }
                    Some(_) => { /* engine quit by it's own */ }
                },
            };
            if kill_needed {
                /* don't be nice */
                match self.process.as_mut().unwrap().kill() {
                    Err(_) => eprintln!("Failed to kill process..."),
                    Ok(_) => {}
                }
            }
            self.process = None;
        }
    }

    pub fn send_command(&mut self, cmd: String) -> Option<String> {
        if self.process.is_none() {
            eprintln!("Engine was not started.");
            return None;
        }
        let process = self.process.as_mut().unwrap();
        let engine_stdin = process.stdin.as_mut().unwrap();
        match engine_stdin.write((String::from(cmd.trim()) + "\n").as_bytes()) {
            Err(_) => {
                eprintln!("Failed to pass command");
                return None;
            }
            Ok(_) => {}
        }
        drop(engine_stdin);

        let mut engine_stdout = BufReader::new(process.stdout.as_mut().unwrap());
        let mut output_line = String::new();

        match engine_stdout.read_line(&mut output_line) {
            Err(_) => {
                eprintln!("Failed to read output from engine");
                return None;
            }
            Ok(_) => {
                return Some(String::from(output_line.trim()));
            }
        }
    }
}

impl HexPlayer for HexPlayerUXI {
    fn next_move(&mut self, position: &HexPosition) -> Option<Location> {
        let command = String::from("next_move ") + &UXI::position_to_uxi(position);
        match self.send_command(command) {
            None => None,
            Some(response) => {
                let m_str: Vec<_> = response.split(",").collect();
                if m_str.len() != 2 {
                    eprintln!("Expected move as r,c format: \"{}\"", response);
                    return None;
                }
                let r = match m_str[0].parse::<usize>() {
                    Err(_) => {
                        eprintln!("Failed to parse row index");
                        return None;
                    }
                    Ok(row) => row,
                };
                let c = match m_str[1].parse::<usize>() {
                    Err(_) => {
                        eprintln!("Failed to parse column index");
                        return None;
                    }
                    Ok(column) => column,
                };
                return Some((r, c));
            }
        }
    }
}

pub struct UXI {}
impl UXI {
    pub fn compare_engines(
        engine1_filename: &std::path::Path,
        engine2_filename: &std::path::Path,
        number_of_games: usize,
        working_dir: &std::path::Path,
    ) -> Option<String> {
        let mut rng = rand::thread_rng();
        // let engine1_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
        // let engine2_errfile = working_dir.join("errlog" + rng.gen::<u64>().to_string());
        let mut engine1 = HexPlayerUXI::new(engine1_filename /*, engine1_errfile*/);
        let mut engine2 = HexPlayerUXI::new(engine2_filename /*, engine2_errfile*/);

        let engine1_started = engine1.start();
        let engine2_started = engine2.start();

        if !engine1_started || !engine2_started {
            if engine1_started {
                engine1.stop();
            }
            if engine2_started {
                engine2.stop();
            }
            eprintln!("Failed to start engines.");
            return None;
        }

        let mut player1_wins = 0;
        let mut player2_wins = 0;
        for _ in 0..number_of_games {
            let starting_player = match rng.gen::<bool>() {
                true => Color::Red,
                false => Color::Blue,
            };
            let mut game = HexGame::new(starting_player, &mut engine1, &mut engine2);
            match game.play_until_over() {
                None => {}
                Some(winner) => match winner {
                    Color::Red => player1_wins += 1,
                    Color::Blue => player2_wins += 1,
                },
            };
        }
        println!("Engines results:");
        println!(
            "{}/{} : {}",
            player1_wins,
            number_of_games,
            engine1_filename.display()
        );
        println!(
            "{}/{} : {}",
            player2_wins,
            number_of_games,
            engine2_filename.display()
        );

        engine1.stop();
        engine2.stop();

        if player1_wins == player2_wins {
            return None;
        }
        if player1_wins > player2_wins {
            return Some(String::from(engine1_filename.to_str().unwrap()));
        }
        return Some(String::from(engine2_filename.to_str().unwrap()));
    }

    pub fn run_engine(player: &mut dyn HexPlayer) {
        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("failed to read input");
            let args: Vec<_> = line.split_whitespace().collect();

            if args.is_empty() {
                continue;
            }
            match args[0] {
                "next_move" => {
                    if args.len() != 3 {
                        eprintln!("Expected position and color for next_move command.");
                        continue;
                    }
                    let pos_str = args[1];
                    let color_str = args[2];
                    match UXI::uxi_to_position(pos_str, color_str) {
                        None => {
                            eprintln!("Failed to parse position.");
                            continue;
                        }
                        Some(pos) => {
                            match player.next_move(&pos) {
                                None => println!("error"),
                                Some(m) => println!("{},{}", m.0, m.1),
                            };
                        }
                    }
                }
                "quit" => {
                    break;
                }
                unknown_cmd => {
                    eprintln!("Unknown command: {}", unknown_cmd);
                }
            }
        }
    }

    fn position_to_uxi(position: &HexPosition) -> String {
        let mut s = String::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                s += match position.get_tile(r, c) {
                    Hexagon::Empty => "e",
                    Hexagon::Full(color) => match color {
                        Color::Red => "r",
                        Color::Blue => "b",
                    },
                };
            }
        }
        return s
            + " "
            + match position.get_turn() {
                Color::Red => "r",
                Color::Blue => "b",
            };
    }

    fn uxi_to_position(pos_str: &str, color_str: &str) -> Option<HexPosition> {
        let mut board: [[Hexagon; BOARD_SIZE]; BOARD_SIZE] =
            [[Hexagon::Empty; BOARD_SIZE]; BOARD_SIZE];
        let mut i = 0;
        for tile in pos_str.chars() {
            if i >= BOARD_SIZE * BOARD_SIZE {
                eprintln!("Too many chars in position string");
                return None;
            }
            board[i / BOARD_SIZE][i % BOARD_SIZE] = match tile {
                'e' => Hexagon::Empty,
                'r' => Hexagon::Full(Color::Red),
                'b' => Hexagon::Full(Color::Blue),
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
            "r" => Color::Red,
            "b" => Color::Blue,
            unknown_player => {
                eprintln!("Unknown player: {}", unknown_player);
                return None;
            }
        };
        return Some(HexPosition::from_board(board, player));
    }
}
