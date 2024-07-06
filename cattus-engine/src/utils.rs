use crate::game::model;
use std::env;
use std::path::Path;

#[cfg(feature = "python")]
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
    if cfg!(feature = "python") {
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

    let mps_available = {
        #[cfg(feature = "python")]
        {
            pyo3::Python::with_gil(|py| {
                let locals = pyo3::types::PyDict::new_bound(py);
                py.run_bound(
                    "
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
        #[cfg(not(feature = "python"))]
        {
            false
        }
    };

    let model_impl = if cfg!(feature = "python") && mps_available {
        model::ImplType::Py
    } else if cfg!(feature = "tract") {
        model::ImplType::Tract
    } else if cfg!(feature = "ort") {
        model::ImplType::Ort
    } else if cfg!(feature = "python") {
        model::ImplType::Py
    } else {
        panic!("No model implementation available");
    };

    match model_impl {
        model::ImplType::Py => {}
        model::ImplType::Tract => {}
        model::ImplType::Ort => {
            #[cfg(feature = "ort")]
            ort_lib::init()
                .with_execution_providers(vec![
                    ort_lib::TensorRTExecutionProvider::default().build(),
                    ort_lib::CUDAExecutionProvider::default().build(),
                    ort_lib::CoreMLExecutionProvider::default().build(),
                ])
                .commit()
                .unwrap();
        }
    }
    model::set_impl_type(model_impl);
}
