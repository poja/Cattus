use ndarray::ArrayD;
use std::{path::Path, sync::Mutex};

#[cfg(feature = "python")]
use {pyo3::prelude::*, pyo3::types::PyDict, pyo3::Py, pyo3::Python};

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum ImplType {
    Py,
    Tract,
    Ort,
}
static IMPL_TYPE: Mutex<Option<ImplType>> = Mutex::new(None);
pub fn set_impl_type(impl_type: ImplType) {
    log::info!("Setting model implementation: {:?}", impl_type);
    *IMPL_TYPE.lock().unwrap() = Some(impl_type);
}

enum ModelImpl {
    #[cfg(feature = "python")]
    Py(Py<PyDict>),
    #[cfg(feature = "tract")]
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "ort")]
    Ort(ort_lib::Session),
}

pub struct Model {
    model: ModelImpl,
}
impl Model {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let impl_type = IMPL_TYPE.lock().unwrap().expect("impl_type not set");
        let model = match impl_type {
            #[cfg(feature = "python")]
            ImplType::Py => {
                let model = Python::with_gil(|py| {
                    let locals = PyDict::new_bound(py);
                    locals
                        .set_item("path", path.with_extension("pt").to_str().unwrap())
                        .unwrap();
                    py.run_bound(
                        r#"
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
model = torch.load(path, map_location=device)
model.eval()
                            "#,
                        None,
                        Some(&locals),
                    )
                    .unwrap();
                    locals.into()
                });
                ModelImpl::Py(model)
            }
            #[cfg(feature = "tract")]
            ImplType::Tract => {
                let model = tract_onnx::onnx()
                    .model_for_path(path.with_extension("onnx"))
                    .unwrap()
                    .into_optimized()
                    .unwrap()
                    .into_runnable()
                    .unwrap();
                ModelImpl::Tract(model)
            }
            #[cfg(feature = "ort")]
            ImplType::Ort => {
                let model = ort_lib::SessionBuilder::new()
                    .unwrap()
                    .with_optimization_level(ort_lib::GraphOptimizationLevel::Level3)
                    .unwrap()
                    .commit_from_file(path.with_extension("onnx"))
                    .unwrap();
                ModelImpl::Ort(model)
            }
            #[cfg(not(all(feature = "python", feature = "tract", feature = "ort")))]
            unsupported_type => panic!("Unsupported impl_type: {:?}", unsupported_type),
        };
        Self { model }
    }

    pub fn run(&self, inputs: Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        match &self.model {
            #[cfg(feature = "python")]
            ModelImpl::Py(locals) => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| {
                        let shape = input.shape().to_vec();
                        let data = input.into_raw_vec();
                        (shape, data)
                    })
                    .collect::<Vec<_>>();
                let outputs = Python::with_gil(|py| {
                    let locals = locals.bind(py);
                    locals.set_item("inputs", inputs).unwrap();
                    py.run_bound(
                        r#"
with torch.no_grad():
    inputs = [torch.tensor(data).view(shape).to(device) for shape, data in inputs]
    outputs = model(*inputs)
    outputs = [output.detach().cpu().numpy() for output in outputs]
    outputs = [(o.shape, o.flatten()) for o in outputs]
                            "#,
                        None,
                        Some(locals),
                    )
                    .unwrap();
                    locals
                        .get_item("outputs")
                        .unwrap()
                        .unwrap()
                        .extract::<Vec<(Vec<usize>, Vec<f32>)>>()
                        .unwrap()
                });
                outputs
                    .into_iter()
                    .map(|(shape, data)| ArrayD::from_shape_vec(shape, data).unwrap())
                    .collect()
            }
            #[cfg(feature = "tract")]
            ModelImpl::Tract(model) => {
                let inputs = TVec::from_vec(
                    inputs
                        .into_iter()
                        .map(Tensor::from)
                        .map(TValue::from)
                        .collect(),
                );
                let mut outputs = model.run(inputs).unwrap();
                (0..outputs.len())
                    .map(|output_idx| {
                        outputs
                            .remove(output_idx)
                            .into_arc_tensor()
                            .into_tensor()
                            .into_array()
                            .unwrap()
                    })
                    .collect()
            }
            #[cfg(feature = "ort")]
            ModelImpl::Ort(model) => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| {
                        ort_lib::SessionInputValue::from(
                            ort_lib::DynValue::try_from(input).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let inputs: &[ort_lib::SessionInputValue] = &inputs;
                let mut outputs = model.run(inputs).unwrap();
                model
                    .outputs
                    .iter()
                    .map(|output_def| {
                        outputs
                            .remove(output_def.name.as_str())
                            .unwrap()
                            .try_extract_tensor()
                            .unwrap()
                            .into_owned()
                    })
                    .collect::<Vec<_>>()
            }
        }
    }
}
