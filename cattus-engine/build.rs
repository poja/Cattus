fn main() {
    if cfg!(feature = "ort") && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }
}
