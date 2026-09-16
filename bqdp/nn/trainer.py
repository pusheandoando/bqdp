# bqdp/nn/trainer.py
import random
from typing import Optional

import torch
from torch import nn

from .objective import full_softmax_loss, sampled_softmax_loss





class OnlineTrainer:
    def __init__(self, network, config, device, generator: torch.Generator):
        self.network = network
        self.config = config
        self.device = device
        self.generator = generator
        self.rng = random.Random(config.seed)
        self.optimizer = torch.optim.AdamW(
            network.parameters(),
            lr = config.learning_rate,
            weight_decay = config.weight_decay,
        )
        self.last_loss = 0.0


    def step(self, history) -> float:
        inputs, targets = history.training_pairs(
            self.config.batch_size,
            self.config.max_sequence_length,
            self.rng,
        )
        
        if not inputs:
            return self.last_loss

        batch_inputs = torch.tensor(inputs, dtype=torch.long, device=self.device)
        batch_targets = torch.tensor(targets, dtype=torch.long, device=self.device)

        self.network.train()
        accumulated = 0.0
        
        for _ in range(self.config.steps_per_update):
            loss = self._compute_loss(batch_inputs, batch_targets)
            if loss is None:
                return self.last_loss
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            accumulated += float(loss.detach())

        self.last_loss = accumulated / self.config.steps_per_update
        
        return self.last_loss


    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        hidden = self.network(inputs)
        
        if self.config.negative_samples > 0:
            return sampled_softmax_loss(
                hidden,
                self.network.item_embedding.weight,
                targets,
                self.config.negative_samples,
                self.generator,
            )
            
        return full_softmax_loss(self.network.full_logits(hidden), targets)


    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "last_loss": self.last_loss}


    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.last_loss = state["last_loss"]