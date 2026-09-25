import stat

from fastapi.testclient import TestClient

from laya_steering.dashboard import create_app
from laya_steering.generation import infer_task
from laya_steering.providers import StaticProvider
from laya_steering.studio import ProviderConfig, StudioService


def test_studio_shell_and_assets(tmp_path):
    client = TestClient(create_app(tmp_path / "experiments.sqlite3"))
    shell = client.get("/")
    assert shell.status_code == 200
    assert "Laya Studio" in shell.text
    assert "<pre>" not in shell.text
    assert client.get("/create").status_code == 200
    assert client.get("/runs").status_code == 200
    assert "New specialization" in client.get("/static/app.js").text
    assert "--teal" in client.get("/static/app.css").text
    assert client.get("/api/runs").json() == []


def test_env_connection_is_private_and_reloadable(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("UNRELATED=value\n", encoding="utf-8")
    service = StudioService(tmp_path / "studio", tmp_path / "lab.sqlite3", env_path)

    public = service.save_connection(
        ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4.1-mini",
            api_key="test-secret",
        )
    )

    assert public == {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4.1-mini",
        "has_api_key": True,
    }
    assert "test-secret" not in str(public)
    assert "UNRELATED=value" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    provider = service.provider(
        ProviderConfig(base_url=public["base_url"], model=public["model"], api_key="")
    )
    assert provider.api_key == "test-secret"


def test_task_labels_can_be_inferred():
    response = {
        "task": {
            "name": "inferred_task",
            "description": "Classify a request using labels inferred by the provider.",
            "domain": "test",
            "decision": {
                "type": "choice",
                "labels": ["ALLOW", "BLOCK"],
                "class_descriptions": {"ALLOW": "safe", "BLOCK": "unsafe"},
            },
        }
    }
    task, record = infer_task(
        StaticProvider([response]), "Infer appropriate policy labels", None, 3
    )
    assert task.decision.labels == ["ALLOW", "BLOCK"]
    assert "Infer 2 to 6" in record.prompt
