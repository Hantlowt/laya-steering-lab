"""Local OpenAI-compatible deterministic server for exercising the Studio UI."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def response_for(prompt: str) -> dict:
    if "Infer a rigorous classification policy" in prompt:
        return {
            "task": {
                "name": "studio_support_priority",
                "description": "Classify support requests by whether immediate intervention is required.",
                "domain": "customer_support",
                "decision": {
                    "type": "choice",
                    "labels": ["NORMAL", "URGENT"],
                    "question": "Does this request require immediate human intervention?",
                    "class_descriptions": {
                        "NORMAL": "Routine help without immediate operational or customer harm.",
                        "URGENT": "Outage, security incident, or active material financial harm.",
                    },
                },
                "semantic_concepts": ["service availability", "security", "customer impact"],
                "positive_concepts": ["active outage", "data exposure"],
                "negative_concepts": ["documentation", "cosmetic preference"],
                "policy": "Choose URGENT only for active widespread harm, compromise, or irreversible loss.",
                "metadata": {},
            }
        }
    match = re.search(
        r"Create exactly (\d+).*?\b(SPECIALIZATION|VALIDATION|HIDDEN|PARAPHRASE|HARD)\b",
        prompt,
        re.DOTALL,
    )
    if not match:
        return {"ok": True}
    count, split = int(match.group(1)), match.group(2).lower()
    examples = []
    for index in range(count):
        if split == "paraphrase":
            pair = index // 2
            urgent = pair % 2 == 0
            label = "URGENT" if urgent else "NORMAL"
            text = (
                f"Paraphrase {index}: every production region is unavailable for customers."
                if urgent
                else f"Paraphrase {index}: asks where to update a harmless profile preference."
            )
            pair_id = f"pair-{pair}"
        else:
            urgent = index % 2 == 0
            label = "URGENT" if urgent else "NORMAL"
            text = (
                f"{split} case {index}: checkout is down for all customers and payments fail."
                if urgent
                else f"{split} case {index}: asks how to change a profile icon without impact."
            )
            pair_id = None
        examples.append(
            {"input": text, "label": label, "tags": [split, "mock-e2e"], "pair_id": pair_id}
        )
    return {"examples": examples}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        prompt = payload.get("messages", [{}, {}])[-1].get("content", "")
        content = response_for(prompt)
        body = json.dumps(
            {
                "id": "mock-studio",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 9099), Handler).serve_forever()
