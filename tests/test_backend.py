import numpy as np

from laya_studio.backends import LayaBackend, task_question


def test_fake_backend_contract(binary_task, fake_backend):
    assert isinstance(fake_backend, LayaBackend)
    embedded = fake_backend.embed(["one", "two"])
    assert embedded.shape == (2, 16)
    assert embedded.dtype == np.float32
    results = fake_backend.predict_batch(["one", "two"], binary_task)
    assert len(results) == 2
    assert all(abs(sum(x["probabilities"].values()) - 1) < 1e-6 for x in results)


def test_question_preserves_label_order(binary_task):
    question = task_question(binary_task, ["URGENT", "NORMAL"])
    assert list(question["decision"]["criteria"]) == ["URGENT", "NORMAL"]
