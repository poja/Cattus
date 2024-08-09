fn main() {
    if cfg!(feature = "onnx-ort") && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }
}
