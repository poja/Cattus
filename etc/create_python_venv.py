import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent


def main() -> None:
    if os.name == "nt":  # Windows
        bin_dir = ROOT_DIR / ".venv" / "Scripts"
        uv_exe = "uv.exe"
    else:
        bin_dir = ROOT_DIR / ".venv" / "bin"
        uv_exe = "uv"
    if not (bin_dir / uv_exe).exists():
        subprocess.check_call(["uv", "venv", "-p", "3.12", "--seed"], cwd=ROOT_DIR)
        subprocess.check_call([bin_dir / "pip", "install", "uv"], cwd=ROOT_DIR)
    subprocess.check_call(
        [bin_dir / "python", ROOT_DIR / "etc" / "install_python_requirements.py"],
        cwd=ROOT_DIR,
    )


if __name__ == "__main__":
    main()
