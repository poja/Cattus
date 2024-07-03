use std::env;
use std::path::Path;

pub trait Builder<T>: Sync + Send {
    fn build(&self) -> T;
}

#[derive(Copy, Clone)]
pub enum Device {
    Cpu,
    Gpu,
}

pub fn init_python() {
    let venv_path = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(".venv");
    let py_packages_path = if cfg!(target_os = "windows") {
        venv_path.join("Lib").join("site-packages")
    } else {
        venv_path
            .join("lib")
            .join("python3.12")
            .join("site-packages")
    };
    env::set_var("PYTHONPATH", py_packages_path.to_str().unwrap());
}
