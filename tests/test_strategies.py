import numpy as np
import pytest

from laya_studio.strategies import ContrastiveVectorStrategy, PrototypeStrategy, normalize


def test_normalize_and_mean_prototypes(binary_task, rows, fake_backend):
    spec = [x for x in rows if x.split == "specialization"]
    fitted = PrototypeStrategy("mean").fit(binary_task, spec, [], fake_backend)
    assert fitted.arrays["prototypes"].shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(fitted.arrays["prototypes"], axis=1), 1, atol=1e-6)
    predictions = fitted.predict([x.input for x in spec])
    assert all(set(x["probabilities"]) == {"NORMAL", "URGENT"} for x in predictions)


def test_medoid_is_an_observed_embedding(binary_task, rows, fake_backend):
    spec = [x for x in rows if x.split == "specialization"]
    fitted = PrototypeStrategy("medoid").fit(binary_task, spec, [], fake_backend)
    observed = normalize(fake_backend.embed([x.input for x in spec]))
    assert all(
        any(np.allclose(center, row) for row in observed) for center in fitted.arrays["prototypes"]
    )


def test_contrastive_is_difference_of_class_means(binary_task, rows, fake_backend):
    spec = [x for x in rows if x.split == "specialization"]
    validation = [x for x in rows if x.split == "validation"]
    fitted = ContrastiveVectorStrategy().fit(binary_task, spec, validation, fake_backend)
    embeddings = normalize(fake_backend.embed([x.input for x in spec]))
    expected = normalize((embeddings[2:].mean(0) - embeddings[:2].mean(0))[None])[0]
    np.testing.assert_allclose(fitted.arrays["direction"], expected, atol=1e-6)
    assert fitted.artifact.params["strength"] in (0.5, 1.0, 2.0, 4.0)


def test_contrastive_rejects_multiclass(binary_task, rows, fake_backend):
    task = binary_task.model_copy(deep=True)
    task.decision.labels.append("OTHER")
    with pytest.raises(ValueError, match="exactly two"):
        ContrastiveVectorStrategy().fit(task, rows[:4], [], fake_backend)
