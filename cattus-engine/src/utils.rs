pub trait Builder<T>: Sync + Send {
    fn build(&self) -> T;
}

#[derive(Copy, Clone)]
pub enum Device {
    Cpu,
    Cuda,
    Mps,
}

pub fn init_globals(device: Option<Device>) {
    ort::init()
        .with_execution_providers(match device {
            Some(Device::Cpu) => vec![],
            Some(Device::Cuda) => vec![
                ort::TensorRTExecutionProvider::default().build(),
                ort::CUDAExecutionProvider::default().build(),
            ],
            Some(Device::Mps) => vec![ort::CoreMLExecutionProvider::default().build()],
            None => vec![
                ort::TensorRTExecutionProvider::default().build(),
                ort::CUDAExecutionProvider::default().build(),
                ort::CoreMLExecutionProvider::default().build(),
            ],
        })
        .commit()
        .unwrap();
}
