/// A copy of rand_dist::dirichlet of version 0.4.0
/// Remove once we bump to 0.6.0
#[allow(dead_code)]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) mod dirichlet;

#[cfg(feature = "torch-python")]
pub(crate) mod python;

use crate::game::model;
use std::env;
use std::path::Path;

#[cfg(feature = "torch-python")]
use pyo3::types::PyAnyMethods;

pub trait Builder<T>: Sync + Send {
    fn build(&self) -> T;
}

#[derive(Copy, Clone)]
pub enum Device {
    Cpu,
    Cuda,
    Mps,
}

pub fn init_globals() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Stdout)
        .init();

    if cfg!(feature = "torch-python") {
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
        unsafe { env::set_var("PYTHONPATH", py_packages_path.to_str().unwrap()) };
    }

    let mps_available = {
        #[cfg(feature = "torch-python")]
        {
            pyo3::Python::attach(|py| {
                let locals = pyo3::types::PyDict::new(py);
                py.run(
                    c"
import torch
mps_available = torch.backends.mps.is_available()
                    ",
                    None,
                    Some(&locals),
                )
                .unwrap();
                locals.get_item("mps_available").unwrap().extract().unwrap()
            })
        }
        #[cfg(not(feature = "torch-python"))]
        {
            false
        }
    };

    let model_impl = if cfg!(feature = "torch-python") && mps_available {
        model::ImplType::Py
    } else if cfg!(feature = "onnx-tract") {
        model::ImplType::Tract
    } else if cfg!(feature = "onnx-ort") {
        model::ImplType::Ort
    } else if cfg!(feature = "torch-python") {
        model::ImplType::Py
    } else {
        panic!("No model implementation available");
    };

    match model_impl {
        model::ImplType::Py => {}
        model::ImplType::Tract => {}
        model::ImplType::Ort => {
            #[cfg(feature = "onnx-ort")]
            ort::init()
                .with_execution_providers(vec![
                    ort::execution_providers::TensorRTExecutionProvider::default().build(),
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
                .commit()
                .unwrap();
        }
    }
    model::set_impl_type(model_impl);
}
