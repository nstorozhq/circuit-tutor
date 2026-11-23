"""Replay frozen follow-up cases through FastAPI and real GigaChat, in temporary SQLite.

State checks are automatic; correctness/relevance of the text requires a separate review.
Run from agent: .venv/bin/python tests/evaluate_followups.py --output tests/followup_results.json
"""
import argparse
import copy
import json
import sys
import tempfile
import time
import hashlib
import fcntl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import create_app
from app.model import GigaChat
from app.rag import SYSTEM_PROMPT, VERIFIER_PROMPT, SUPPORT_PLAN_PROMPT, SUPPORT_PROMPT, GROUNDING_PROMPT
from app.knowledge import COURSE
from app.tutor import current_question, new_state


def evaluate(output, suite=None, only=(), pause=0):
    cases = json.loads(Path(__file__).with_name("followup_cases.json").read_text())
    model = GigaChat()
    if not model.enabled:
        raise SystemExit("Configure GIGACHAT_CREDENTIALS for live evaluation.")
    results = []
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__).resolve().parents[1] / "app").glob("*.py")}
    kb = Path(__file__).resolve().parents[1] / "knowledge_base" / "lesson.json"
    source_hashes["knowledge_base/lesson.json"] = hashlib.sha256(kb.read_bytes()).hexdigest()
    calls = []
    real_complete = model.complete
    async def trace(prompt, payload):
        response = await real_complete(prompt, payload)
        kind = next((name for name, value in (
            ("extract", SYSTEM_PROMPT), ("verify", VERIFIER_PROMPT), ("plan", SUPPORT_PLAN_PROMPT),
            ("select", SUPPORT_PROMPT), ("check", GROUNDING_PROMPT),
        ) if value == prompt), "other")
        calls.append({"kind": kind, "response": response,
                      "source_ids": [source["id"] for source in payload.get("sources", [])],
                      "evidence_chunks": payload.get("evidence_chunks")})
        return response
    model.complete = trace
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(Path(tmp) / "check.sqlite3", model)) as client:
        for case in cases:
            if suite and case["suite"] != suite or only and case["id"] not in only:
                continue
            index = next(i for i, item in enumerate(COURSE) if item["id"] == case["case"])
            before = new_state()
            before.update(index=index, phase=case.get("phase", COURSE[index]["stage"]),
                          operation=case["step"], spoken_named=not case.get("naming", False))
            if case.get("pending"):
                before["pending"] = {"key": "correct", "prompt": current_question(before)["prompt"]}
            history = [{"id": f"fixture-{i}", **item} for i, item in enumerate(case.get("history", []))]
            session = client.post("/api/sessions").json()
            sid = session["id"]
            with client.app.state.store.connect() as db:
                db.execute("UPDATE sessions SET state=?,history=? WHERE id=?", (json.dumps(before), json.dumps(history), sid))
            started = time.monotonic()
            calls.clear()
            rows = []
            for n, text in enumerate([case["text"]] + ([case["followup"]] if case.get("followup") else [])):
                prior = copy.deepcopy(before) if n == 0 else copy.deepcopy(after)
                response = client.post(f"/api/sessions/{sid}/turns", json={
                    "version": n, "message_id": f"check-{n}", "text": text,
                })
                response.raise_for_status()
                result = response.json()
                saved = client.get(f"/api/sessions/{sid}").json()
                _, after, _, _ = client.app.state.store.load(sid, None)
                changed = any(after[key] != prior[key] for key in
                              ("phase", "index", "operation", "spoken_named", "case_correct", "scores"))
                expectation = case["expect"] if n == 0 else case["followup_expect"]
                rows.append({"text": text, "before": prior, "after": copy.deepcopy(after),
                             "reply": result["message"]["text"], "grounding": result["message"].get("grounding"),
                             "state_check": changed == (expectation == "progress"),
                             "state_consistent": result["state"] == saved["state"],
                             "full_pending_preserved": after["pending"] == prior["pending"]})
            row = {**case, "turns": rows, "model_calls": list(calls), "seconds": round(time.monotonic() - started, 2), "content_review": "pending"}
            row["provider_check"] = before["phase"] == "test" or all(turn["grounding"] and turn["grounding"]["status"] == "matched" for turn in rows if turn["text"] != "Подтвердить")
            if "expect_pending" in case:
                row["pending_check"] = bool(rows[0]["after"]["pending"]) == case["expect_pending"]
            results.append(row)
            output.write_text(json.dumps({"model": model.model, "source_hashes": source_hashes, "target": "local FastAPI / real GigaChat / temporary SQLite",
                                          "results": results}, ensure_ascii=False, indent=2))
            print(json.dumps({"id": case["id"], "state_checks": [item["state_check"] for item in rows],
                              "replies": [item["reply"] for item in rows]}, ensure_ascii=False), flush=True)
            if not row["provider_check"]:
                print("Stopped: provider/extraction failure is not a passing dialogue test.", flush=True)
                return 2
            if pause:
                time.sleep(pause)
    return sum(not all(turn["state_check"] and turn["state_consistent"] for turn in row["turns"])
               or row.get("pending_check") is False for row in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("original", "audit"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--pause", type=float, default=0, help="Pause between cases to respect provider quotas")
    args = parser.parse_args()
    with open(Path(tempfile.gettempdir()) / "circuit-tutor-live-evaluation.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another live evaluation is running; wait for it to finish.")
        raise SystemExit(evaluate(args.output, args.suite, args.only, args.pause))
