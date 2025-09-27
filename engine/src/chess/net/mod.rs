pub mod common;
pub mod net_trivial;
pub mod net_two_headed;

mod net_trivial_test;

#[cfg(feature = "stockfish")]
pub mod stockfish;
