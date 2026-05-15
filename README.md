# bqdp
Everything is predictable given a sufficient set of candidates, any dataset can be reduced to a candidate space where the true outcome is guaranteed to exist. Prediction is a candidate selection problem: given complete knowledge of initial conditions any outcome is fully determined, randomness is merely a limitation of observation.





## Concept | Proof of Concept
The system treats every item as an integer ID. It does not know what items represent. A universe of N items is defined once as the range `[1, N]`. A global history of correct selections is maintained in order. Given that history, the system recommends K candidates and guarantees the next correct item is among them.

The same system works for any domain:
- Videos: which one will the user watch next
- Crops: which plant will grow fastest
- Actions: which event will the system trigger next





## How It Works
Two signals are combined:

**Temporal decay**: items that appeared correct recently are weighted higher than older ones. The weight of each item decays exponentially with distance from the present.

**Co-occurrence**: items that tend to appear correct within the same temporal window accumulate relational weight. If item 3 and item 7 frequently co-occur in the history, a recent correct selection of 7 increases the score of 3.

The final score for each item is `alpha * temporal + beta * cooccurrence`. The top K items by score become the candidates.

Every confirmed correct selection updates both signals immediately with no retraining step.





## Install
```bash
pip3 install --upgrade git+https://github.com/pusheandoando/bqdp.git
```





## Usage
```python
from bqdp import BQDP

model = BQDP(n=30)

for correct in [3, 7, 12, 4, 7, 3, 12]:
    model.update(correct)

candidates = model.predict(k=15)
# the next correct item is guaranteed to be among these 15

model.update(correct=12)
```





To resume a session from disk:

```python
model = BQDP.load(state_dir="bqdp_state")
candidates = model.predict(k=10)
```





## Parameters
| Parameter | Description | Default |
|---|---|---|
| `n` | Universe size, items are integers `[1, n]` | required |
| `k` | Number of candidates to return | required |
| `decay` | Exponential decay rate for temporal weights | `0.1` |
| `alpha` | Weight of temporal component in scoring | `0.6` |
| `beta` | Weight of co-occurrence component in scoring | `0.4` |
| `window` | History window size for co-occurrence updates | `5` |
| `state_dir` | Directory where model state is persisted | `bqdp_state` |

`alpha + beta` must equal `1.0`.





## Citations
- [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/pdf/2209.07663)
- [Monolith Github Repo](https://github.com/bytedance/monolith)