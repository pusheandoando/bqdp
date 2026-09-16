# bqdp/nn/objective.py
from typing import Optional

import torch
from torch.nn import functional as functional_ops





def full_softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
    flat_targets = targets.reshape(-1)
    
    if not bool((flat_targets != 0).any()):
        return None

    flat_logits = logits.reshape(-1, logits.size(-1))
    
    padding = torch.zeros(flat_logits.size(-1), dtype=torch.bool, device=flat_logits.device)
    padding[0] = True
    
    flat_logits = flat_logits.masked_fill(padding, float("-inf"))
    
    return functional_ops.cross_entropy(flat_logits, flat_targets, ignore_index=0)


def sampled_softmax_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, num_negatives: int, generator: torch.Generator) -> Optional[torch.Tensor]:
    valid = targets != 0
    
    if not bool(valid.any()):
        return None

    anchors = hidden[valid]
    positives = targets[valid]

    # negatives are shared across the batch to keep the logit matrix at [valid, negatives]
    negatives = torch.randint(1, weight.size(0), (num_negatives,), generator=generator)
    negatives = negatives.to(anchors.device)

    positive_logits = (anchors * weight[positives]).sum(dim=-1, keepdim=True)
    negative_logits = torch.matmul(anchors, weight[negatives].t())
    collision = negatives.unsqueeze(0) == positives.unsqueeze(1)
    negative_logits = negative_logits.masked_fill(collision, float("-inf"))

    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    return functional_ops.cross_entropy(logits, labels)