pub trait Builder<T>: Sync + Send {
    fn build(&self) -> T;
}
