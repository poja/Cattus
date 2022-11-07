pub trait Callback<Args> {
    fn call(&self, args: Args);
}
