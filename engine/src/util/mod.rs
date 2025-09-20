pub(crate) mod batch;
pub(crate) mod metrics;

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

pub fn init_globals(model_impl: Option<model::ImplType>) {
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

    let model_impl = model_impl.unwrap_or_else(|| {
        if cfg!(feature = "executorch") {
            model::ImplType::Executorch
        } else if cfg!(feature = "torch-python") && mps_available {
            model::ImplType::TorchPy
        } else if cfg!(feature = "onnx-tract") {
            model::ImplType::OnnxTract
        } else if cfg!(feature = "onnx-ort") {
            model::ImplType::OnnxOrt
        } else if cfg!(feature = "torch-python") {
            model::ImplType::TorchPy
        } else {
            panic!("No model implementations available in this build");
        }
    });

    match model_impl {
        model::ImplType::TorchPy => {}
        model::ImplType::Executorch => {
            #[cfg(feature = "executorch")]
            {
                unsafe extern "C" fn cattus_executorch_emit_log(
                    #[allow(unused)] timestamp: executorch_sys::executorch_timestamp_t,
                    level: executorch_sys::executorch_pal_log_level,
                    filename: *const ::core::ffi::c_char,
                    #[allow(unused)] function: *const ::core::ffi::c_char,
                    line: usize,
                    message: *const ::core::ffi::c_char,
                    length: usize,
                ) {
                    let filename = unsafe {
                        assert!(!filename.is_null());
                        let filename = std::ffi::CStr::from_ptr(filename);
                        filename.to_str().unwrap()
                    };
                    let message = unsafe {
                        assert!(!message.is_null());
                        let bytes = std::slice::from_raw_parts(message.cast(), length);
                        let message = std::ffi::CStr::from_bytes_with_nul_unchecked(bytes);
                        message.to_str().unwrap()
                    };
                    let level = match level {
                        executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_DEBUG => log::Level::Debug,
                        executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_INFO => log::Level::Info,
                        executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_ERROR => log::Level::Error,
                        executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_FATAL => log::Level::Error,
                        executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_UNKNOWN => log::Level::Error,
                    };
                    let logger = log::logger();
                    logger.log(
                        &log::Record::builder()
                            .args(format_args!("{message}"))
                            .level(level)
                            .file(Some(filename))
                            .line(Some(line as u32))
                            .build(),
                    );
                    logger.flush();
                }
                unsafe {
                    executorch_sys::executorch_register_pal(executorch_sys::ExecutorchPalImpl {
                        init: None,
                        abort: None,
                        current_ticks: None,
                        ticks_to_ns_multiplier: None,
                        emit_log_message: Some(cattus_executorch_emit_log),
                        allocate: None,
                        free: None,
                        source_filename: std::ptr::null(),
                    });
                    executorch::platform::pal_init();
                }
            }
        }
        model::ImplType::OnnxTract => {}
        model::ImplType::OnnxOrt => {
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
