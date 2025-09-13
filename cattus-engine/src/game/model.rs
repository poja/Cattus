use ndarray::ArrayD;
use std::path::Path;
use std::sync::Mutex;

#[cfg(feature = "torch-python")]
use {crate::util::python::Unwrapy, pyo3::prelude::*};

#[cfg(feature = "onnx-tract")]
use tract_onnx::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum ImplType {
    TorchPy,
    Executorch,
    OnnxTract,
    OnnxOrt,
}
static IMPL_TYPE: Mutex<Option<ImplType>> = Mutex::new(None);
pub fn set_impl_type(impl_type: ImplType) {
    log::debug!("Setting model implementation: {:?}", impl_type);
    *IMPL_TYPE.lock().unwrap() = Some(impl_type);
}

#[allow(clippy::large_enum_variant)]
enum ModelImpl {
    #[cfg(feature = "torch-python")]
    Py(Py<PyAny>),
    #[cfg(feature = "executorch")]
    Executorch(executorch::module::Module<'static>),
    #[cfg(feature = "onnx-tract")]
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "onnx-ort")]
    Ort {
        model: ort::session::Session,
        output_names: Vec<String>,
    },
}

pub struct Model {
    model: ModelImpl,
}
impl Model {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let impl_type = IMPL_TYPE.lock().unwrap().expect("impl_type not set");
        let model = match impl_type {
            #[cfg(feature = "torch-python")]
            ImplType::TorchPy => {
                let model = Python::attach(|py| {
                    let code = cr#"
import torch
class Model:
    def __init__(self, path):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    def run(self, inputs):
        with torch.no_grad():
            inputs = [torch.from_numpy(input).to(self.device) for input in inputs]
            outputs = self.model(*inputs)
            outputs = [output.detach().cpu().numpy() for output in outputs]
        return outputs
                        "#;
                    let module =
                        PyModule::from_code(py, code, c"py/model.py", c"model").unwrapy(py);

                    let py_class = module.getattr("Model").unwrap();
                    py_class
                        .call1((path.with_extension("jit"),))
                        .unwrapy(py)
                        .into()
                });
                ModelImpl::Py(model)
            }
            #[cfg(feature = "executorch")]
            ImplType::Executorch => {
                let mut model =
                    executorch::module::Module::from_file_path(path.with_extension("pte"));
                model
                    .load(Some(
                        executorch::program::ProgramVerification::InternalConsistency,
                    ))
                    .unwrap();
                model.load_method("forward", None, None).unwrap();
                ModelImpl::Executorch(model)
            }
            #[cfg(feature = "onnx-tract")]
            ImplType::OnnxTract => {
                let model = tract_onnx::onnx()
                    .model_for_path(path.with_extension("onnx"))
                    .unwrap()
                    .into_optimized()
                    .unwrap()
                    .into_runnable()
                    .unwrap();
                ModelImpl::Tract(model)
            }
            #[cfg(feature = "onnx-ort")]
            ImplType::OnnxOrt => {
                let model = ort::session::Session::builder()
                    .unwrap()
                    .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                    .unwrap()
                    .commit_from_file(path.with_extension("onnx"))
                    .unwrap();
                let output_names = model.outputs.iter().map(|o| o.name.clone()).collect();
                ModelImpl::Ort {
                    model,
                    output_names,
                }
            }
            #[cfg(not(all(
                feature = "torch-python",
                feature = "executorch",
                feature = "onnx-tract",
                feature = "onnx-ort"
            )))]
            unsupported_type => panic!("Unsupported impl_type: {:?}", unsupported_type),
        };
        Self { model }
    }

    pub fn run(&mut self, inputs: Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        match &mut self.model {
            #[cfg(feature = "torch-python")]
            ModelImpl::Py(model) => Python::attach(|py| {
                use itertools::Itertools;
                use numpy::{IntoPyArray, PyArrayMethods};

                let inputs = inputs
                    .into_iter()
                    .map(|input| input.into_pyarray(py))
                    .collect::<Vec<_>>();
                let outputs = model.bind(py).call_method1("run", (inputs,)).unwrap();
                outputs
                    .extract::<Vec<Py<numpy::PyArrayDyn<f32>>>>()
                    .unwrap()
                    .into_iter()
                    .map(|o| o.into_bound(py).to_owned_array())
                    .collect_vec()
            }),
            #[cfg(feature = "executorch")]
            ModelImpl::Executorch(model) => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| executorch::tensor::TensorPtr::from_array(input).unwrap())
                    .collect::<Vec<_>>();
                let inputs = inputs
                    .iter()
                    .map(executorch::evalue::EValue::from)
                    .collect::<Vec<_>>();
                let outputs = model.forward(&inputs).unwrap();
                outputs
                    .into_iter()
                    .map(|o| o.as_tensor().into_typed::<f32>().as_array().to_owned())
                    .collect()
            }
            #[cfg(feature = "onnx-tract")]
            ModelImpl::Tract(model) => {
                let inputs = TVec::from_vec(
                    inputs
                        .into_iter()
                        .map(Tensor::from)
                        .map(TValue::from)
                        .collect(),
                );
                let outputs = model.run(inputs).unwrap();
                outputs
                    .into_iter()
                    .map(|o| o.into_tensor().into_array::<f32>().unwrap())
                    .collect()
            }
            #[cfg(feature = "onnx-ort")]
            ModelImpl::Ort {
                model,
                output_names,
            } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| {
                        ort::session::SessionInputValue::from(
                            ort::value::Value::from_array(input).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let inputs: &[ort::session::SessionInputValue] = &inputs;
                let mut outputs = model.run(inputs).unwrap();
                output_names
                    .iter()
                    .map(|output_name| {
                        outputs
                            .remove(output_name)
                            .unwrap()
                            .try_extract_array::<f32>()
                            .unwrap()
                            .into_owned()
                    })
                    .collect::<Vec<_>>()
            }
        }
    }
}
