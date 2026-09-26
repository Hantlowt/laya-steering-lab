# User Sentiment Satisfaction Classifier

Portable Laya specialization artifact (format version 1).

- Base model: `convaiinnovations/laya`
- Strategy: `whitened_prototypes`
- Task: A deterministic policy that assigns one of three sentiment labels—SATISFIED, MIXED, or UNSATISFIED—to a single user-provided sentence based on the expressed attitude toward a product, service, experience, or outcome.
- Decision labels: SATISFIED, MIXED, UNSATISFIED
- License: Apache-2.0

```python
from laya_steering import SpecializedLaya
agent = SpecializedLaya.from_pretrained(".")
result = agent.predict("your input")
```

Laya model weights by Convai Innovations and upstream contributors.
