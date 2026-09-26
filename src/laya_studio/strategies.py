from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .backends import LayaBackend, parse_laya_result, task_question
from .schemas import DecisionType, Example, StrategyArtifact, TaskSpec

EPS = 1e-12


class StrategyNotApplicable(ValueError):
    """The strategy is valid but does not apply to this task schema."""


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, EPS)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(float(temperature), EPS)
    z -= z.max(axis=-1, keepdims=True)
    values = np.exp(z)
    return values / values.sum(axis=-1, keepdims=True)


@dataclass
class FittedStrategy:
    artifact: StrategyArtifact
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    specialization_seconds: float = 0.0
    _predict: Callable[[list[str]], list[dict[str, Any]]] | None = field(default=None, repr=False)

    def predict(self, states: list[str]) -> list[dict[str, Any]]:
        if self._predict is None:
            raise RuntimeError("strategy is not attached to a backend")
        return self._predict(states)


class Strategy(ABC):
    name: str
    decision_component: str

    @abstractmethod
    def fit(
        self,
        task: TaskSpec,
        specialization: list[Example],
        validation: list[Example],
        backend: LayaBackend,
    ) -> FittedStrategy:
        raise NotImplementedError


class BaselineStrategy(Strategy):
    name, decision_component = "baseline", "laya_head"

    def fit(self, task, specialization, validation, backend):
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name, labels=task.decision.labels, decision_component="laya_head"
            ),
            _predict=lambda states: backend.predict_batch(states, task),
        )


class PromptOnlyStrategy(Strategy):
    name, decision_component = "prompt_only", "laya_head"

    def fit(self, task, specialization, validation, backend):
        descriptions = "; ".join(
            f"{label}: {task.decision.class_descriptions.get(label, label)}"
            for label in task.decision.labels
        )
        instruction = (
            f"{task.description}\nApply this policy exactly. {descriptions}. "
            "Resolve boundary cases using the class definitions, not label position."
        )

        def predict(states):
            question = task_question(task, instructions=instruction)
            agent = getattr(backend, "agent", None)
            if agent is None:
                return backend.predict_batch(states, task)
            if hasattr(agent, "predict_batch"):
                raw = agent.predict_batch(states, question)
            else:
                raw = [agent.predict(state, question) for state in states]
            return [parse_laya_result(x, task) for x in raw]

        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="laya_head",
                params={"rewritten_instructions": instruction},
            ),
            _predict=predict,
        )


class PrototypeStrategy(Strategy):
    decision_component = "embedding_classifier"

    def __init__(self, mode: str = "mean", top_k: int = 3, temperature: float = 0.2):
        if mode not in {"mean", "medoid", "nearest", "top_k"}:
            raise ValueError("prototype mode must be mean, medoid, nearest, or top_k")
        self.mode, self.top_k, self.temperature = mode, top_k, temperature
        self.name = "nearest_prototype"

    def fit(self, task, specialization, validation, backend):
        started = time.perf_counter()
        embeddings = normalize(backend.embed([x.input for x in specialization]))
        y = np.array([task.decision.labels.index(x.label) for x in specialization])
        representatives = []
        for index, _label in enumerate(task.decision.labels):
            subset = embeddings[y == index]
            if not len(subset):
                raise ValueError(f"no specialization examples for label {_label!r}")
            centroid = normalize(subset.mean(axis=0, keepdims=True))[0]
            if self.mode == "medoid":
                representatives.append(subset[int(np.argmax(subset @ centroid))])
            else:
                representatives.append(centroid)
        centers = np.stack(representatives)

        def predict(states):
            x = normalize(backend.embed(states))
            if self.mode in {"nearest", "top_k"}:
                similarities = x @ embeddings.T
                if self.mode == "nearest":
                    logits = np.full((len(x), len(task.decision.labels)), -1.0, dtype=np.float32)
                    winners = np.argmax(similarities, axis=1)
                    for row, winner in enumerate(winners):
                        logits[row, y[winner]] = similarities[row, winner]
                else:
                    k = min(self.top_k, len(embeddings))
                    nearest = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
                    logits = np.zeros((len(x), len(task.decision.labels)), dtype=np.float32)
                    for row, indices in enumerate(nearest):
                        for idx in indices:
                            logits[row, y[idx]] += max(float(similarities[row, idx]), 0.0) + EPS
            else:
                logits = x @ centers.T
            return probability_rows(softmax(logits, self.temperature), task.decision.labels)

        arrays = {
            "prototypes": centers,
            "example_embeddings": embeddings,
            "example_labels": y.astype(np.int64),
        }
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="embedding_classifier",
                params={"mode": self.mode, "top_k": self.top_k, "temperature": self.temperature},
                arrays=sorted(arrays),
            ),
            arrays,
            time.perf_counter() - started,
            predict,
        )


class MulticlassCentroidsStrategy(PrototypeStrategy):
    def __init__(self, temperature: float = 0.2):
        super().__init__("mean", temperature=temperature)
        self.name = "multiclass_centroids"


class ContrastiveVectorStrategy(Strategy):
    name, decision_component = "contrastive_vector", "embedding_classifier"

    def __init__(self, strengths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)):
        self.strengths = strengths

    def fit(self, task, specialization, validation, backend):
        if len(task.decision.labels) != 2:
            raise StrategyNotApplicable("contrastive_vector requires exactly two labels")
        started = time.perf_counter()
        emb = normalize(backend.embed([x.input for x in specialization]))
        mask = np.array([x.label == task.decision.labels[1] for x in specialization])
        if not mask.any() or mask.all():
            raise ValueError("contrastive_vector requires both labels")
        direction = normalize((emb[mask].mean(0) - emb[~mask].mean(0))[None, :])[0]
        val_emb = (
            normalize(backend.embed([x.input for x in validation]))
            if validation
            else np.empty((0, len(direction)))
        )
        val_y = np.array([task.decision.labels.index(x.label) for x in validation])
        best_strength, best_score = self.strengths[0], -1.0
        for strength in self.strengths:
            if not len(validation):
                break
            pred = (val_emb @ direction * strength >= 0).astype(int)
            score = float((pred == val_y).mean())
            if score > best_score:
                best_strength, best_score = strength, score

        def predict(states):
            score = normalize(backend.embed(states)) @ direction * best_strength
            probs_pos = 1.0 / (1.0 + np.exp(-score))
            probs = np.column_stack((1.0 - probs_pos, probs_pos))
            return probability_rows(probs, task.decision.labels)

        arrays = {"direction": direction}
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="embedding_classifier",
                params={"strength": best_strength, "validation_accuracy": best_score},
                arrays=["direction"],
            ),
            arrays,
            time.perf_counter() - started,
            predict,
        )


class WhitenedPrototypesStrategy(Strategy):
    name, decision_component = "whitened_prototypes", "embedding_classifier"

    def __init__(self, shrinkage: float = 0.1, temperature: float = 0.2):
        self.shrinkage, self.temperature = shrinkage, temperature

    def fit(self, task, specialization, validation, backend):
        started = time.perf_counter()
        emb = backend.embed([x.input for x in specialization]).astype(np.float64)
        mean = emb.mean(0)
        centered = emb - mean
        covariance = centered.T @ centered / max(len(emb) - 1, 1)
        scale = float(np.trace(covariance) / max(len(mean), 1))
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * scale * np.eye(len(mean))
        values, vectors = np.linalg.eigh(covariance)
        whitening = (vectors * (1.0 / np.sqrt(np.maximum(values, 1e-6)))) @ vectors.T
        transformed = normalize(centered @ whitening)
        y = np.array([task.decision.labels.index(x.label) for x in specialization])
        centers = normalize(
            np.stack([transformed[y == i].mean(0) for i in range(len(task.decision.labels))])
        )

        def predict(states):
            x = normalize((backend.embed(states) - mean) @ whitening)
            return probability_rows(softmax(x @ centers.T, self.temperature), task.decision.labels)

        arrays = {
            "mean": mean.astype(np.float32),
            "whitening": whitening.astype(np.float32),
            "prototypes": centers,
        }
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="embedding_classifier",
                params={"shrinkage": self.shrinkage, "temperature": self.temperature},
                arrays=sorted(arrays),
            ),
            arrays,
            time.perf_counter() - started,
            predict,
        )


class ResidualEmbeddingTransformStrategy(Strategy):
    name, decision_component = "residual_embedding_transform", "embedding_classifier"

    def __init__(self, ridge: float = 1.0, temperature: float = 0.2):
        self.ridge, self.temperature = ridge, temperature

    def fit(self, task, specialization, validation, backend):
        started = time.perf_counter()
        x = normalize(backend.embed([e.input for e in specialization])).astype(np.float64)
        label_desc = [
            f"{label} {task.decision.class_descriptions.get(label, '')}"
            for label in task.decision.labels
        ]
        target_centers = normalize(backend.embed(label_desc)).astype(np.float64)
        y = np.stack([target_centers[task.decision.labels.index(e.label)] for e in specialization])
        dim = x.shape[1]
        # Closed-form residual ridge map. No gradients and no Laya parameter updates.
        delta = np.linalg.solve(x.T @ x + self.ridge * np.eye(dim), x.T @ (y - x))
        transform = np.eye(dim) + delta
        centers = normalize(
            np.stack(
                [
                    normalize(x @ transform)[
                        np.array([e.label == label for e in specialization])
                    ].mean(0)
                    for label in task.decision.labels
                ]
            )
        )

        def predict(states):
            z = normalize(normalize(backend.embed(states)) @ transform)
            return probability_rows(softmax(z @ centers.T, self.temperature), task.decision.labels)

        arrays = {
            "transform": transform.astype(np.float32),
            "prototypes": centers.astype(np.float32),
        }
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="embedding_classifier",
                params={"ridge": self.ridge, "temperature": self.temperature},
                arrays=sorted(arrays),
            ),
            arrays,
            time.perf_counter() - started,
            predict,
        )


class MultiVectorStrategy(Strategy):
    name, decision_component = "multi_vector_steering", "embedding_classifier"

    def __init__(self, vectors_per_class: int = 2, temperature: float = 0.2):
        self.vectors_per_class, self.temperature = vectors_per_class, temperature

    def fit(self, task, specialization, validation, backend):
        started = time.perf_counter()
        emb = normalize(backend.embed([x.input for x in specialization]))
        vectors, owners = [], []
        for label_idx, label in enumerate(task.decision.labels):
            subset = emb[np.array([x.label == label for x in specialization])]
            if not len(subset):
                raise ValueError(f"no examples for {label}")
            # Deterministic farthest-point semantic sub-prototypes.
            centers = [subset[int(np.argmax(np.linalg.norm(subset - subset.mean(0), axis=1)))]]
            while len(centers) < min(self.vectors_per_class, len(subset)):
                similarity = subset @ normalize(np.stack(centers)).T
                centers.append(subset[int(np.argmin(similarity.max(axis=1)))])
            assignment = np.argmax(subset @ normalize(np.stack(centers)).T, axis=1)
            for i in range(len(centers)):
                vectors.append(normalize(subset[assignment == i].mean(0, keepdims=True))[0])
                owners.append(label_idx)
        vectors_array, owners_array = np.stack(vectors), np.asarray(owners)

        def predict(states):
            similarity = normalize(backend.embed(states)) @ vectors_array.T
            logits = np.stack(
                [similarity[:, owners_array == i].max(1) for i in range(len(task.decision.labels))],
                axis=1,
            )
            return probability_rows(softmax(logits, self.temperature), task.decision.labels)

        arrays = {"vectors": vectors_array, "owners": owners_array.astype(np.int64)}
        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="embedding_classifier",
                params={
                    "vectors_per_class": self.vectors_per_class,
                    "temperature": self.temperature,
                },
                arrays=sorted(arrays),
            ),
            arrays,
            time.perf_counter() - started,
            predict,
        )


class ActivationSteeringStrategy(Strategy):
    name, decision_component = "activation_steering", "laya_head"

    def __init__(self, layer: int = -1, strengths: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)):
        self.layer, self.strengths = layer, strengths

    def fit(self, task, specialization, validation, backend):
        if not backend.info.supports_activation_steering:
            raise NotImplementedError(f"activation_steering unsupported on {backend.info.name}")
        if len(task.decision.labels) != 2:
            raise StrategyNotApplicable("activation_steering currently requires a binary task")
        started = time.perf_counter()
        emb = backend.embed([x.input for x in specialization])
        positive = np.array([x.label == task.decision.labels[1] for x in specialization])
        vector = normalize((emb[positive].mean(0) - emb[~positive].mean(0))[None, :])[0]
        best, best_accuracy = self.strengths[0], -1.0
        for strength in self.strengths:
            if not validation:
                break
            predictions = backend.predict_steered_batch(
                [x.input for x in validation], task, vector, strength, self.layer
            )
            accuracy = np.mean([p["label"] == e.label for p, e in zip(predictions, validation)])
            if accuracy > best_accuracy:
                best, best_accuracy = strength, float(accuracy)

        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=task.decision.labels,
                decision_component="laya_head",
                params={
                    "layer": self.layer,
                    "strength": best,
                    "validation_accuracy": best_accuracy,
                },
                arrays=["task_vector"],
            ),
            {"task_vector": vector},
            time.perf_counter() - started,
            lambda states: backend.predict_steered_batch(states, task, vector, best, self.layer),
        )


class PairwiseRankingStrategy(Strategy):
    name, decision_component = "pairwise_ranking", "pairwise_laya_head"

    def fit(self, task, specialization, validation, backend):
        labels = task.decision.labels

        def predict(states):
            totals = np.zeros((len(states), len(labels)), dtype=np.float32)
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    pair_task = task.model_copy(deep=True)
                    pair_task.decision.type = DecisionType.choice
                    pair_task.decision.labels = [labels[i], labels[j]]
                    out = backend.predict_batch(states, pair_task)
                    for row, value in enumerate(out):
                        totals[row, labels.index(value["label"])] += 1.0
            probs = totals / np.maximum(totals.sum(1, keepdims=True), 1.0)
            return probability_rows(probs, labels)

        return FittedStrategy(
            StrategyArtifact(
                strategy=self.name,
                labels=labels,
                decision_component="pairwise_laya_head",
                params={"aggregation": "Copeland"},
            ),
            _predict=predict,
        )


def probability_rows(probs: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    return [
        {"label": labels[int(np.argmax(row))], "probabilities": dict(zip(labels, map(float, row)))}
        for row in probs
    ]


STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "baseline": BaselineStrategy,
    "prompt_only": PromptOnlyStrategy,
    "prompt": PromptOnlyStrategy,
    "nearest_prototype": PrototypeStrategy,
    "prototype": PrototypeStrategy,
    "contrastive_vector": ContrastiveVectorStrategy,
    "contrastive": ContrastiveVectorStrategy,
    "multiclass_centroids": MulticlassCentroidsStrategy,
    "centroids": MulticlassCentroidsStrategy,
    "whitened_prototypes": WhitenedPrototypesStrategy,
    "whitened": WhitenedPrototypesStrategy,
    "residual_embedding_transform": ResidualEmbeddingTransformStrategy,
    "residual": ResidualEmbeddingTransformStrategy,
    "activation_steering": ActivationSteeringStrategy,
    "steering": ActivationSteeringStrategy,
    "multi_vector_steering": MultiVectorStrategy,
    "multivector": MultiVectorStrategy,
    "pairwise_ranking": PairwiseRankingStrategy,
    "pairwise": PairwiseRankingStrategy,
}


def create_strategy(name: str) -> Strategy:
    if name.startswith("activation_steering@"):
        try:
            return ActivationSteeringStrategy(layer=int(name.rsplit("@", 1)[1]))
        except ValueError as exc:
            raise ValueError(
                "activation layer must be an integer, e.g. activation_steering@-1"
            ) from exc
    try:
        return STRATEGIES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from exc
