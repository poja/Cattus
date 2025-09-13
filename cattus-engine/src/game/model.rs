use ndarray::ArrayD;
use std::path::Path;
use std::sync::Mutex;

#[cfg(feature = "python")]
use {pyo3::prelude::*, pyo3::Py, pyo3::Python};

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
    log::debug!("Setting model implementation: {:?}", impl_type);
    *IMPL_TYPE.lock().unwrap() = Some(impl_type);
}

#[allow(clippy::large_enum_variant)]
enum ModelImpl {
    #[cfg(feature = "python")]
    Py(Py<PyAny>),
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
                    let code = r#"
import torch
class Model:
    def __init__(self, path):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = torch.load(path, map_location=self.device)
        self.model.eval()

    def run(self, inputs):
        with torch.no_grad():
            inputs = [torch.tensor(data).view(shape).to(self.device) for shape, data in inputs]
            outputs = self.model(*inputs)
            outputs = [output.detach().cpu().numpy() for output in outputs]
            outputs = [(o.shape, o.flatten()) for o in outputs]
        return outputs
                        "#;
                    let module = PyModule::from_code_bound(py, code, "py/model.py", "model")
                        .map_err(
                            // print the familiar python stack trace
                            |err| err.print_and_set_sys_last_vars(py),
                        )
                        .unwrap();

                    let py_class = module.getattr("Model").unwrap();
                    py_class
                        .call1((path.with_extension("pt"),))
                        .map_err(
                            // print the familiar python stack trace
                            |err| err.print_and_set_sys_last_vars(py),
                        )
                        .unwrap()
                        .into()
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
            ModelImpl::Py(model) => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| {
                        let shape = input.shape().to_vec();
                        let data = input.into_raw_vec();
                        (shape, data)
                    })
                    .collect::<Vec<_>>();
                let outputs = Python::with_gil(|py| {
                    let py_fn = model.getattr(py, "run").unwrap();
                    py_fn
                        .call1(py, (inputs,))
                        .map_err(
                            // print the familiar python stack trace
                            |err| err.print_and_set_sys_last_vars(py),
                        )
                        .unwrap()
                        .extract::<Vec<(Vec<usize>, Vec<f32>)>>(py)
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

                let mut outputs = (0..outputs.len())
                    .rev()
                    .map(|output_idx| {
                        outputs
                            .remove(output_idx)
                            .into_arc_tensor()
                            .into_tensor()
                            .into_array()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                outputs.reverse();
                outputs
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
