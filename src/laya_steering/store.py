from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  suite_path TEXT NOT NULL,
  base_model TEXT NOT NULL,
  base_revision TEXT,
  backend TEXT NOT NULL,
  seed INTEGER NOT NULL,
  environment_json TEXT NOT NULL,
  git_commit TEXT,
  display_name TEXT
);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT NOT NULL REFERENCES runs(id),
  task_name TEXT NOT NULL,
  domain TEXT NOT NULL,
  strategy TEXT NOT NULL,
  decision_component TEXT NOT NULL,
  strategy_params_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  timings_json TEXT NOT NULL,
  kept INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, task_name, strategy)
);
CREATE TABLE IF NOT EXISTS predictions (
  run_id TEXT NOT NULL,
  task_name TEXT NOT NULL,
  strategy TEXT NOT NULL,
  example_id TEXT NOT NULL,
  split TEXT NOT NULL,
  input_text TEXT NOT NULL,
  expected TEXT NOT NULL,
  predicted TEXT NOT NULL,
  probabilities_json TEXT NOT NULL,
  baseline_predicted TEXT,
  PRIMARY KEY (run_id, strategy, example_id)
);
CREATE TABLE IF NOT EXISTS specializations (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  task_name TEXT NOT NULL,
  strategy TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_filters ON results(domain, strategy);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(base_model);
"""


class ExperimentStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        run_columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
        if "display_name" not in run_columns:
            db.execute("ALTER TABLE runs ADD COLUMN display_name TEXT")
        result_columns = {row[1] for row in db.execute("PRAGMA table_info(results)")}
        if "kept" not in result_columns:
            db.execute("ALTER TABLE results ADD COLUMN kept INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create_run(self, record: dict[str, Any]) -> None:
        fields = (
            "id",
            "created_at",
            "status",
            "suite_path",
            "base_model",
            "base_revision",
            "backend",
            "seed",
            "environment_json",
            "git_commit",
            "display_name",
        )
        with self.connect() as db:
            db.execute(
                f"INSERT INTO runs ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                tuple(record.get(key) for key in fields),
            )

    def set_status(self, run_id: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def set_run_name(self, run_id: str, name: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE runs SET display_name=? WHERE id=?", (name.strip()[:120], run_id))

    def set_result_kept(self, run_id: str, task: str, strategy: str, kept: bool) -> None:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE results SET kept=? WHERE run_id=? AND task_name=? AND strategy=?",
                (int(kept), run_id, task, strategy),
            )
            if not cursor.rowcount:
                raise KeyError(f"unknown result {run_id}:{task}:{strategy}")

    def add_result(self, row: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO results
                (run_id,task_name,domain,strategy,decision_component,strategy_params_json,
                 metrics_json,timings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["run_id"],
                    row["task_name"],
                    row["domain"],
                    row["strategy"],
                    row["decision_component"],
                    json.dumps(row["strategy_params"], sort_keys=True),
                    json.dumps(row["metrics"], sort_keys=True),
                    json.dumps(row["timings"], sort_keys=True),
                ),
            )

    def add_predictions(self, rows: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.executemany(
                """INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r["run_id"],
                        r["task_name"],
                        r["strategy"],
                        r["example_id"],
                        r["split"],
                        r["input"],
                        r["expected"],
                        r["predicted"],
                        json.dumps(r["probabilities"], sort_keys=True),
                        r.get("baseline_predicted"),
                    )
                    for r in rows
                ],
            )

    def add_specialization(
        self, specialization_id: str, run_id: str, task_name: str, strategy: str, artifact_path: str
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO specializations VALUES (?, ?, ?, ?, ?, ?)",
                (specialization_id, run_id, task_name, strategy, artifact_path, now_iso()),
            )

    def specialization(self, specialization_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                """SELECT s.*, r.base_model, r.base_revision, r.backend, r.suite_path
                FROM specializations s LEFT JOIN runs r ON r.id=s.run_id WHERE s.id=?""",
                (specialization_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown specialization {specialization_id}")
        return dict(row)

    def run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"unknown run {run_id}")
            results = db.execute(
                "SELECT * FROM results WHERE run_id=? ORDER BY task_name,strategy", (run_id,)
            ).fetchall()
        return {"run": dict(run), "results": [_decode_result(dict(x)) for x in results]}

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT r.*, COUNT(x.task_name) result_count,
                MIN(x.task_name) primary_task, SUM(COALESCE(x.kept,0)) kept_count
                FROM runs r LEFT JOIN results x ON x.run_id=r.id
                GROUP BY r.id ORDER BY r.created_at DESC"""
            ).fetchall()
        return [dict(x) for x in rows]

    def mistakes(self, run_id: str, task: str, strategy: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM predictions WHERE run_id=? AND task_name=? AND strategy=?
                AND expected != predicted ORDER BY example_id""",
                (run_id, task, strategy),
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["probabilities"] = json.loads(value.pop("probabilities_json"))
            result.append(value)
        return result


def _decode_result(row: dict[str, Any]) -> dict[str, Any]:
    row["strategy_params"] = json.loads(row.pop("strategy_params_json"))
    row["metrics"] = json.loads(row.pop("metrics_json"))
    row["timings"] = json.loads(row.pop("timings_json"))
    row["kept"] = bool(row.get("kept", 0))
    return row


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
