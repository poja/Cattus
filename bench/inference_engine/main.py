import argparse
import copy
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import yaml
from cattus_train import Config
from cattus_train.chess import Chess
from cattus_train.config import (
    ExecutorchConfig,
    OnnxOrtConfig,
    OnnxTractConfig,
    TorchPyConfig,
)
from cattus_train.self_play import compile_selfplay_exe

CATTUS_TOP = Path(__file__).parent.parent.parent.resolve()
SELF_PLAY_CRATE_TOP = CATTUS_TOP / "training" / "self-play"
GAME = Chess()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CATTUS_TOP / "training" / "config" / "chess_dev.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = Config(**yaml.safe_load(args.config.read_text()))
    if cfg.device == "auto":
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"
    cfg.engine.mcts.sim_num = 4
    cfg.engine.threads = 1
    cfg.engine.batch_size = 1
    cfg.self_play.games_num = 2

    inference_configs = [
        ExecutorchConfig(backend="none"),
        ExecutorchConfig(backend="xnnpack"),
        # ExecutorchConfig(backend="mps"),
        TorchPyConfig(),
        OnnxTractConfig(),
        OnnxOrtConfig(),
    ]

    summary = []
    for inference_config in inference_configs:
        with tempfile.TemporaryDirectory() as tempdir:
            print("\n# Benchmarking inference config:", inference_config)

            current_cfg = copy.deepcopy(cfg)
            current_cfg.engine.inference = inference_config
            current_cfg.self_play.engine_overrides.pop("inference", None)

            model_path = Path(tempdir) / "model"
            export_model(current_cfg, model_path)

            executable = compile_selfplay_exe("chess", inference_config, debug=False)
            t0 = time.time()
            score = bench_selfplay(executable, model_path, current_cfg)
            t1 = time.time()

            summary.append((inference_config, score, t1 - t0))

    print("\n # Summary:")
    for inference_config, score, duration in summary:
        print("cfg:", inference_config)
        print("score:", score)
        print("duration:", duration)
        print()


def bench_selfplay(executable: Path, model_path: Path, cfg: Config) -> float:
    from cattus_train.train_process import temperature_policy_to_str

    with tempfile.TemporaryDirectory() as tempdir_:
        tempdir = Path(tempdir_)
        summary_file = tempdir / "summary.json"
        data_entries_dir = tempdir / "data_entries"
        engine_cfg = cfg.self_play_engine_cfg()
        subprocess.check_call(
            [
                executable,
                f"--model1-path={model_path}",
                f"--model2-path={model_path}",
                f"--games-num={cfg.self_play.games_num}",
                f"--out-dir1={data_entries_dir}",
                f"--out-dir2={data_entries_dir}",
                f"--summary-file={summary_file}",
                f"--sim-num={engine_cfg.mcts.sim_num}",
                f"--batch-size={engine_cfg.batch_size}",
                f"--explore-factor={engine_cfg.mcts.explore_factor}",
                f"--temperature-policy={temperature_policy_to_str(engine_cfg.mcts.temperature_policy)}",
                f"--prior-noise-alpha={engine_cfg.mcts.prior_noise_alpha}",
                f"--prior-noise-epsilon={engine_cfg.mcts.prior_noise_epsilon}",
                f"--threads={engine_cfg.threads}",
                f"--device={cfg.device}",
                f"--cache-size={engine_cfg.mcts.cache_size}",
            ],
            cwd=SELF_PLAY_CRATE_TOP,
        )

        with open(summary_file, "r") as f:
            summary = json.load(f)
    return summary["metrics"]["model.run_duration"]


def export_model(cfg: Config, path: Path):
    from cattus_train.self_play import export_model as export_model_impl

    model = GAME.create_model(cfg.model.type, cfg.model.__dict__.copy())

    engine_cfg = cfg.self_play_engine_cfg()
    self_play_input_shape = GAME.model_input_shape(cfg.model.type)
    self_play_input_shape = (engine_cfg.batch_size,) + self_play_input_shape[1:]
    export_model_impl(model, path, engine_cfg.inference, self_play_input_shape)


if __name__ == "__main__":
    main()
