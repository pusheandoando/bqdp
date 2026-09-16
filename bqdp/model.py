# bqdp/model.py
import os
import json
from typing import Dict, List, Optional

import torch

from .config import BQDPConfig
from .router import resolve_state_dir
from .nn import OnlineTrainer, SequenceRecommender, resolve_device
from .core import (
    ConformalCalibrator, Cooccurrence, History, MarkovChain, Scorer, StreamMetrics, Universe
)





_CONFIG_FILE = "config.json"
_STATE_FILE = "state.pt"





class BQDP:
    def __init__(self, n: Optional[int]=None, state_dir: Optional[str]=None, config: Optional[BQDPConfig]=None, **overrides):
        if config is None:
            if n is None:
                raise ValueError("either n or config must be provided")
            
            config = BQDPConfig.build(n, **overrides)
        elif overrides:
            raise ValueError("overrides cannot be combined with an explicit config")

        self.config = config
        self.state_dir = resolve_state_dir() if state_dir is None else resolve_state_dir(state_dir)
        self.device = resolve_device(config.device)

        torch.manual_seed(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed)

        self.universe = Universe(config.n)
        self.history = History(
            self.universe.embedding_size,
            decay = config.decay,
            capacity = config.replay_capacity,
        )
        self.cooccurrence = Cooccurrence(
            self.universe.embedding_size,
            window = config.window,
            decay = config.cooccurrence_decay,
            max_neighbors = config.max_neighbors,
        )
        self.markov = MarkovChain(
            self.universe.embedding_size,
            order = config.markov_order,
            discount = config.markov_discount,
            max_contexts = config.markov_max_contexts,
            max_successors = config.markov_max_successors,
        )
        self.scorer = Scorer(
            {
                "sequence": config.model_weight,
                "markov": config.markov_weight,
                "recency": config.temporal_weight,
                "cooccurrence": config.cooccurrence_weight,
            },
            smoothing = config.smoothing,
            temperature = config.mixture_temperature,
            share = config.mixture_share,
            floor = config.mixture_floor,
        )
        self.conformal = ConformalCalibrator(
            config.conformal_alpha,
            config.conformal_step_size,
            config.conformal_capacity,
        )
        self.metrics = StreamMetrics()
        self.network = SequenceRecommender(
            self.universe.embedding_size,
            config.embedding_dim,
            config.num_layers,
            config.num_heads,
            config.feedforward_dim,
            config.dropout,
            config.max_sequence_length,
        ).to(self.device)
        self.trainer = OnlineTrainer(self.network, config, self.device, self.generator)

        self._expert_cache: Optional[Dict[str, torch.Tensor]] = None
        self._distribution_cache: Optional[torch.Tensor] = None
        self._updates_since_save = 0


    def predict(self, k: int) -> List[int]:
        if k < 1:
            raise ValueError("k must be >= 1")
        
        if k > self.universe.n:
            raise ValueError("k cannot exceed universe size")
        
        return self.scorer.top_k(self._distribution(), k)


    def predict_set(self, alpha: Optional[float]=None, max_size: Optional[int]=None) -> List[int]:
        limit = self.universe.n if max_size is None else min(max_size, self.universe.n)
        
        if limit < 1:
            raise ValueError("max_size must be >= 1")
        
        return self.conformal.prediction_set(self._distribution(), limit, alpha)


    def update(self, correct: int) -> Dict[str, float]:
        self.universe.require(correct)

        # the candidate set is produced before the observation so evaluation stays honest
        experts = self._experts()
        distribution = self._distribution()
        candidates = self.conformal.prediction_set(distribution, self.universe.n)
        covered = self.conformal.observe(distribution, correct)
        
        self.metrics.observe(
            distribution,
            correct,
            min(self.config.evaluation_k, self.universe.n),
            len(candidates),
            covered,
        )
        self.scorer.observe(experts, correct)

        self.markov.update(self.history.recent(self.config.markov_order), correct)
        self.cooccurrence.update(self.history.recent(self.config.window), correct)
        self.history.append(correct)
        self._expert_cache = None
        self._distribution_cache = None

        loss = self.trainer.step(self.history)
        self._maybe_save()

        return {
            "loss": loss,
            "covered": float(covered),
            "set_size": float(len(candidates)),
        }


    def evaluate(self) -> Dict[str, float]:
        report = self.metrics.snapshot()
        report["conformal_alpha"] = self.conformal.effective_alpha
        report["conformal_coverage"] = self.conformal.coverage

        for name, weight in self.scorer.weights.items():
            report[f"weight_{name}"] = weight

        for name, loss in self.scorer.mean_log_loss.items():
            report[f"log_loss_{name}"] = loss

        return report


    def save(self) -> str:
        os.makedirs(self.state_dir, exist_ok=True)
        
        with open(os.path.join(self.state_dir, _CONFIG_FILE), "w") as handle:
            json.dump(self.config.to_dict(), handle, indent=4)
        
        torch.save(self.state_dict(), os.path.join(self.state_dir, _STATE_FILE))
        
        self._updates_since_save = 0
        
        return self.state_dir


    def state_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "trainer": self.trainer.state_dict(),
            "history": self.history.state_dict(),
            "markov": self.markov.state_dict(),
            "cooccurrence": self.cooccurrence.state_dict(),
            "scorer": self.scorer.state_dict(),
            "conformal": self.conformal.state_dict(),
            "metrics": self.metrics.state_dict(),
        }


    def load_state_dict(self, state: dict) -> None:
        self.network.load_state_dict(state["network"])
        self.trainer.load_state_dict(state["trainer"])
        self.history.load_state_dict(state["history"])
        self.markov.load_state_dict(state["markov"])
        self.cooccurrence.load_state_dict(state["cooccurrence"])
        self.scorer.load_state_dict(state["scorer"])
        self.conformal.load_state_dict(state["conformal"])
        self.metrics.load_state_dict(state["metrics"])
        self._expert_cache = None
        self._distribution_cache = None


    @classmethod
    def load(cls, state_dir: Optional[str]=None) -> "BQDP":
        resolved = resolve_state_dir() if state_dir is None else resolve_state_dir(state_dir)
        config_path = os.path.join(resolved, _CONFIG_FILE)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"no bqdp state found in {resolved}")

        with open(config_path) as handle:
            config = BQDPConfig.from_dict(json.load(handle))

        instance = cls(
            config = config,
            state_dir = state_dir
        )
        state = torch.load(
            os.path.join(resolved, _STATE_FILE),
            map_location = instance.device,
            weights_only = True,
        )
        
        instance.load_state_dict(state)
        
        return instance


    def _experts(self) -> Dict[str, torch.Tensor]:
        if self._expert_cache is None:
            self._expert_cache = self.scorer.normalize(self._raw_experts())
        
        return self._expert_cache


    def _distribution(self) -> torch.Tensor:
        if self._distribution_cache is None:
            self._distribution_cache = self.scorer.blend(self._experts())
        
        return self._distribution_cache


    def _raw_experts(self) -> Dict[str, torch.Tensor]:
        return {
            "sequence": self._sequence_distribution(),
            "markov": self.markov.distribution(self.history.recent(self.config.markov_order)),
            "recency": self.history.temporal_weights(),
            "cooccurrence": self.cooccurrence.scores(self.history.recent(self.config.window)),
        }


    def _sequence_distribution(self) -> torch.Tensor:
        sequence = self.history.tail(self.config.max_sequence_length)
        inputs = torch.tensor([sequence], dtype=torch.long, device=self.device)

        self.network.eval()
        
        with torch.no_grad():
            hidden = self.network(inputs)
            logits = self.network.full_logits(hidden[:, -1, :]).squeeze(0)

        return torch.softmax(logits.detach().to("cpu"), dim=-1)


    def _maybe_save(self) -> None:
        if self.config.autosave_interval <= 0:
            return
        
        self._updates_since_save += 1
        
        if self._updates_since_save >= self.config.autosave_interval:
            self.save()