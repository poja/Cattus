pub mod chess;
pub mod hex;
pub mod ttt;

use std::path::Path;

use crate::self_play::DataEntry;

pub trait DataSerializer<Game: cattus::game::Game>: Sync + Send {
    fn serialize_data_entry(&self, entry: DataEntry<Game>, filename: &Path) -> std::io::Result<()>;
}
