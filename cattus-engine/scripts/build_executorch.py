import argparse
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE_DIR = Path(__file__).parent.parent.resolve()
EXECUTORCH_DIR = CRATE_DIR / "third-party" / "executorch"


def main():
    parser = argparse.ArgumentParser("Build executorch from source")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing executorch directory before cloning",
    )
    parser.add_argument(
        "--no-new-venv",
        action="store_true",
        help="Create a new virtual environment for the building process",
    )
    args = parser.parse_args()

    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])

    if not args.no_new_venv:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            if os.name == "nt":  # Windows
                bin_dir = tmpdir / ".venv" / "Scripts"
                uv_exe = "uv.exe"
            else:
                bin_dir = tmpdir / ".venv" / "bin"
                uv_exe = "uv"
            if not (bin_dir / uv_exe).exists():
                subprocess.check_call(
                    ["uv", "venv", "-p", "3.12", "--seed"], cwd=tmpdir
                )
                subprocess.check_call([bin_dir / "pip", "install", "uv"], cwd=tmpdir)
            subprocess.check_call(
                [
                    bin_dir / "python",
                    __file__,
                    *(["--clean"] if args.clean else []),
                    "--no-new-venv",
                ],
            )

    if args.clean:
        if EXECUTORCH_DIR.exists():
            shutil.rmtree(EXECUTORCH_DIR)

    clone_executorch()

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "-r",
            EXECUTORCH_DIR / "requirements-dev.txt",
            "torch==2.8.0",
            "--torch-backend",
            "cpu",
        ]
    )
    build_executorch()


def clone_executorch():
    if not EXECUTORCH_DIR.exists():
        EXECUTORCH_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.7.0",
                "https://github.com/pytorch/executorch.git",
                ".",
            ],
            cwd=EXECUTORCH_DIR,
        )

        if platform.system() == "Darwin":
            # Clone coremltools repo
            # Required on apple when EXECUTORCH_BUILD_DEVTOOLS=ON
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "8.3",
                    "https://github.com/apple/coremltools.git",
                ],
                cwd=EXECUTORCH_DIR / "backends" / "apple" / "coreml" / "scripts",
            )

    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=EXECUTORCH_DIR
    )
    subprocess.check_call(
        ["git", "submodule", "sync", "--recursive"], cwd=EXECUTORCH_DIR
    )


def build_executorch():
    build_dir = EXECUTORCH_DIR / "build"
    if not build_dir.exists():
        build_dir.mkdir()
    subprocess.check_call(
        [
            "cmake",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
            "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
            "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
            "-DEXECUTORCH_ENABLE_LOGGING=OFF",
            "-DEXECUTORCH_BUILD_PORTABLE_OPS=OFF",
            "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
            "-DEXECUTORCH_BUILD_MPS=OFF",
            "-DEXECUTORCH_BUILD_XNNPACK=OFF",
            "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF",
            "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF",
            "-DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF",
            "-DEXECUTORCH_BUILD_DEVTOOLS=OFF",
            "-DEXECUTORCH_ENABLE_EVENT_TRACER=OFF",
            "..",
        ],
        cwd=build_dir,
    )

    subprocess.check_call(
        ["cmake", "--build", build_dir, "-j" + str(multiprocessing.cpu_count() + 1)],
        cwd=EXECUTORCH_DIR,
    )


if __name__ == "__main__":
    main()
