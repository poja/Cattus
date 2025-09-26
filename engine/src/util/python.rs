use pyo3::prelude::*;

pub(crate) trait Unwrapy<T> {
    fn unwrapy(self, py: Python) -> T;
}

impl<T> Unwrapy<T> for PyResult<T> {
    fn unwrapy(self, py: Python) -> T {
        self.inspect_err(|err| err.print_and_set_sys_last_vars(py)).unwrap()
    }
}
