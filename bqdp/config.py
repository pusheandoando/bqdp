# bqdp/config.py
from dataclasses import asdict, dataclass, fields





_ALIASES = {
    "alpha": "temporal_weight",
    "beta": "cooccurrence_weight",
}





@dataclass
class BQDPConfig:
    n: int

    embedding_dim: int=64
    num_layers: int=2
    num_heads: int=2
    feedforward_dim: int=256
    dropout: float=0.2
    max_sequence_length: int=128

    learning_rate: float=1e-3
    weight_decay: float=1e-2
    grad_clip: float=1.0
    batch_size: int=32
    steps_per_update: int=1
    negative_samples: int=-1
    replay_capacity: int=200000

    decay: float=0.1
    window: int=5
    cooccurrence_decay: float=0.0
    max_neighbors: int=64

    markov_order: int=4
    markov_discount: float=0.75
    markov_max_contexts: int=200000
    markov_max_successors: int=64

    model_weight: float=0.25
    markov_weight: float=0.25
    temporal_weight: float=0.25
    cooccurrence_weight: float=0.25
    smoothing: float=1e-3

    mixture_temperature: float=1.0
    mixture_share: float=0.005
    mixture_floor: float=1e-4

    conformal_alpha: float=0.05
    conformal_step_size: float=0.01
    conformal_capacity: int=4096

    evaluation_k: int=10
    autosave_interval: int=100
    device: str="auto"
    seed: int=0


    def __post_init__(self):
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.num_heads < 1 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be >= 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.steps_per_update < 1:
            raise ValueError("steps_per_update must be >= 1")
        if self.replay_capacity < self.max_sequence_length:
            raise ValueError("replay_capacity must be >= max_sequence_length")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.markov_order < 1:
            raise ValueError("markov_order must be >= 1")
        if not 0.0 <= self.markov_discount < 1.0:
            raise ValueError("markov_discount must be in [0.0, 1.0)")
        if self.markov_max_contexts < 1:
            raise ValueError("markov_max_contexts must be >= 1")
        if self.markov_max_successors < 1:
            raise ValueError("markov_max_successors must be >= 1")
        if self.mixture_temperature <= 0.0:
            raise ValueError("mixture_temperature must be > 0.0")
        if not 0.0 <= self.mixture_share < 1.0:
            raise ValueError("mixture_share must be in [0.0, 1.0)")
        if not 0.0 <= self.mixture_floor < 0.25:
            raise ValueError("mixture_floor must leave room for every expert")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must be in (0.0, 1.0)")
        if self.evaluation_k < 1:
            raise ValueError("evaluation_k must be >= 1")
        if self.negative_samples < 0:
            # a full softmax stays affordable only while the universe is small
            self.negative_samples = 1024 if self.n > 4096 else 0


    @classmethod
    def build(cls, n: int, **overrides) -> "BQDPConfig":
        known = {field.name for field in fields(cls)}
        resolved = {}
        
        for key, value in overrides.items():
            name = _ALIASES.get(key, key)
            
            if name not in known:
                raise TypeError(f"unknown configuration option: {key}")
            
            if value is None:
                continue
            
            resolved[name] = value
        
        return cls(n=n, **resolved)


    @classmethod
    def from_dict(cls, payload: dict) -> "BQDPConfig":
        known = {field.name for field in fields(cls)}
        
        return cls(**{key: value for key, value in payload.items() if key in known})


    def to_dict(self) -> dict:
        return asdict(self)