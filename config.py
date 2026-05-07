"""
config.py — Single source of truth for all hyperparameters.
Override any value via environment variables or pass a dict to get_config().
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json, os


@dataclass
class DataConfig:
    n_samples: int        = 124
    n_features: int       = 50        # raw features per modality
    n_qubits: int         = 8         #6         # PCA target & circuit width
    n_classes: int        = 2
    test_size: float      = 0.15
    val_size: float       = 0.15
    noise_level: float    = 0.1    #0.25
    missing_rate: float   = 0.0    #0.05      # synthetic missing value rate
    random_seed: int      = 42


@dataclass
class QuantumConfig:
    n_qubits: int         = 8    #6
    n_layers: int         = 3    #3         # variational layers
    encoding: str         = "amplitude"   # "angle" | "amplitude" | "iqp"
    entanglement: str     = "circular"    # "circular" | "full" | "linear"
    diff_method: str      = "backprop"


@dataclass
class FusionConfig:
    embed_dim: int        = 8   #6
    n_heads: int          = 2
    dropout: float        = 0.15
    use_modality_dropout: bool = True   # randomly drop a modality during training


@dataclass
class TrainConfig:
    epochs: int           = 20   #60
    batch_size: int       = 16   #32
    lr: float             = 3e-3
    weight_decay: float   = 1e-4
    label_smoothing: float = 0.05
    patience: int         = 15
    grad_clip: float      = 1.0
    warmup_epochs: int    = 5
    n_folds: int          = 5          # stratified K-fold
    use_mixed_precision: bool = False  # set True if CUDA available


@dataclass
class Config:
    data:    DataConfig    = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    fusion:  FusionConfig  = field(default_factory=FusionConfig)
    train:   TrainConfig   = field(default_factory=TrainConfig)

    use_quantum: bool = True         #used for ablation
    use_transformer: bool = True    #used for ablation
    
    output_dir: str        = "outputs"
    log_file:   str        = "outputs/run.log"

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.data    = DataConfig(**d["data"])
        cfg.quantum = QuantumConfig(**d["quantum"])
        cfg.fusion  = FusionConfig(**d["fusion"])
        cfg.train   = TrainConfig(**d["train"])
        cfg.output_dir = d["output_dir"]
        cfg.log_file   = d["log_file"]
        return cfg


def get_config() -> Config:
    cfg = Config()
    # keep quantum & fusion in sync with data
    cfg.quantum.n_qubits = cfg.data.n_qubits
    cfg.fusion.embed_dim = cfg.data.n_qubits
    return cfg
