from pathlib import Path
from typing import Literal, Optional

from pydantic.dataclasses import Field, dataclass


@dataclass(config={"extra": "allow"}, kw_only=True)
class ModelConfig:
    type: str
    base: Path | str = "[none]"


@dataclass(config={"extra": "forbid"}, kw_only=True)
class MctsConfig:
    sim_num: int
    explore_factor: float
    prior_noise_alpha: float
    prior_noise_epsilon: float
    cache_size: int = 0


@dataclass(config={"extra": "forbid"}, kw_only=True)
class SelfPlayConfig:
    games_num: int
    batch_size: int
    temperature_policy: list[list[int | float]]
    threads: int


@dataclass(config={"extra": "forbid"}, kw_only=True)
class TrainingConfig:
    batch_size: int
    learning_rate: list[list[float]]
    use_train_data_across_runs: bool = False
    threads: Optional[int] = 1
    latest_data_entries: int
    iteration_data_entries: int

    def __post_init__(self):
        if self.latest_data_entries < self.iteration_data_entries:
            raise ValueError("latest_data_entries must be >= iteration_data_entries")


@dataclass(config={"extra": "forbid"}, kw_only=True)
class ExecutorchConfig:
    engine: Literal["executorch"]
    backend: Literal["none", "xnnpack"] = "none"


@dataclass(config={"extra": "forbid"}, kw_only=True)
class TorchPyConfig:
    engine: Literal["torch-py"]


@dataclass(config={"extra": "forbid"}, kw_only=True)
class OnnxTractConfig:
    engine: Literal["onnx-tract"]


@dataclass(config={"extra": "forbid"}, kw_only=True)
class OnnxOrtConfig:
    engine: Literal["onnx-ort"]


InferenceConfig = ExecutorchConfig | TorchPyConfig | OnnxTractConfig | OnnxOrtConfig


@dataclass(config={"extra": "forbid"}, kw_only=True)
class ModelCompareConfig:
    games_num: int
    temperature_policy: list[list[int | float]]
    threads: int
    switching_winning_threshold: float = Field(ge=0.5, le=1.0)
    warning_losing_threshold: float = Field(ge=0.5, le=1.0)


@dataclass(config={"extra": "forbid"}, kw_only=True)
class Config:
    working_area: Path
    games_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None
    game: str
    model: ModelConfig
    self_play: SelfPlayConfig
    model_compare: ModelCompareConfig
    model_num: int = 1
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    iterations: int
    mcts: MctsConfig
    training: TrainingConfig
    inference: InferenceConfig = Field(discriminator="engine", default=None)
    debug: bool
