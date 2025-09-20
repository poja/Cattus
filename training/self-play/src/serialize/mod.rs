pub mod chess;
pub mod hex;
pub mod ttt;

use std::path::Path;

use cattus::game::common::IGame;

use crate::self_play::DataEntry;

pub trait DataSerializer<Game: IGame>: Sync + Send {
    fn serialize_data_entry(&self, entry: DataEntry<Game>, filename: &Path) -> std::io::Result<()>;
}
