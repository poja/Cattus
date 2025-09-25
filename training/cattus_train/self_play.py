import os
import subprocess
import threading
import warnings
from pathlib import Path

import executorch.exir
import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from cattus_train.config import ExecutorchConfig, InferenceConfig, OnnxOrtConfig, OnnxTractConfig, TorchPyConfig

CATTUS_TOP = Path(__file__).parent.parent.parent.resolve()
SELF_PLAY_CRATE_TOP = CATTUS_TOP / "training" / "self-play"


# For some reason, onnx.export is not thread-safe, so we need to lock it
ONNX_EXPORT_LOCK = threading.RLock()


def compile_selfplay_exe(game: str, cfg: InferenceConfig, debug: bool = False) -> Path:
    env = os.environ.copy()
    match cfg:
        case TorchPyConfig():
            features = ["torch-python"]
        case ExecutorchConfig():
            features = ["executorch"]
            match cfg.backend:
                case "none":
                    pass
                case "xnnpack":
                    env["CATTUS_XNNPACK"] = "1"
                case "mps":
                    env["CATTUS_MPS"] = "1"
                case _:
                    raise ValueError(f"Unsupported executorch backend: {cfg.backend}")
        case OnnxTractConfig():
            features = ["onnx-tract"]
        case OnnxOrtConfig():
            features = ["onnx-ort"]
    self_play_exec_name = f"{game}_self_play_runner"
    profile = "dev" if debug else "release"
    subprocess.check_call(
        [
            "cargo",
            "build",
            f"--profile={profile}",
            f"--features={','.join(features)}",
            "-q",
            f"--bin={self_play_exec_name}",
        ],
        cwd=SELF_PLAY_CRATE_TOP,
        env=env,
    )

    build_dir = "debug" if debug else "release"
    return SELF_PLAY_CRATE_TOP / "target" / build_dir / self_play_exec_name


def export_model_for_selfplay(
    model: nn.Module,
    model_path: Path,
    cfg: InferenceConfig,
    input_shape: tuple[int, ...],
):
    was_training = model.training
    if was_training:
        model.eval()
    with torch.no_grad():
        _export_model_for_selfplay_impl(model, model_path, cfg, input_shape)
    if was_training:
        model.train()


def _export_model_for_selfplay_impl(
    model: nn.Module,
    model_path: Path,
    cfg: InferenceConfig,
    input_shape: tuple[int, ...],
) -> Path:
    sample_input = torch.randn(input_shape)

    match cfg:
        case TorchPyConfig():  # torch.jit
            traced_model = torch.jit.trace(model, sample_input)
            torch.jit.save(traced_model, model_path.with_suffix(".jit"))

        case ExecutorchConfig():  # executorch
            with warnings.catch_warnings():
                exported_model = torch.export.export(model, (sample_input,))
                edge_program = executorch.exir.to_edge(exported_model)

                match cfg.backend:
                    case "none":
                        pass
                    case "xnnpack":
                        edge_program = edge_program.to_backend(XnnpackPartitioner())
                    case "mps":
                        from executorch.backends.apple.mps.partition import MPSPartitioner
                        # from executorch.backends.apple.mps import MPSBackend

                        compile_specs = []
                        # edge_program = edge_program.to_backend(MPSBackend())
                        edge_program = edge_program.to_backend(MPSPartitioner(compile_specs=compile_specs))

                        # use_fp16 = True
                        # compile_specs = [CompileSpec("use_fp16", bytes([use_fp16]))]
                        # use_partitioner = True
                        # if use_partitioner:
                        #     et_model = et_model.to_backend(MPSPartitioner(compile_specs=compile_specs))
                        # else:
                        #     et_model = to_backend(MPSBackend.__name__, et_model.exported_program(), compile_specs)
                        #     et_model = export_to_edge(et_model, (self_play_sample_input,))
                        # et_program = et_model.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))
                        # et_program = to_edge_transform_and_lower(
                        #     et_model,
                        #     # partitioner=[MPSPartitioner(compile_specs=[CompileSpec("use_fp16", bytes([True]))])],
                        #     partitioner=[XnnpackPartitioner()]
                        # ).to_executorch()
                    case _:
                        raise ValueError(f"Unsupported executorch backend: {cfg.backend}")

                # print(f"Lowered graph:\n{edge_program.exported_program().graph}")
                et_program = edge_program.to_executorch()

                with open(model_path.with_suffix(".pte"), "wb") as f:
                    et_program.write_to_file(f)

        case OnnxOrtConfig() | OnnxTractConfig():  # onnx
            with ONNX_EXPORT_LOCK:
                torch.onnx.export(
                    model,
                    sample_input,
                    model_path.with_suffix(".onnx"),
                    verbose=False,
                    input_names=["planes"],
                    output_names=["policy", "value"],
                )
        case _:
            raise ValueError(f"Unsupported inference engine: {cfg.inference}")
