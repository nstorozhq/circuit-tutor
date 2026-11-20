"""Run the fixed dialogue scenarios against the real configured GigaChat."""
import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # Loads the same local configuration as normal startup.
from app.knowledge import COURSE
from app.model import GigaChat
from app.tutor import new_state, respond


async def evaluate(output, only=()):
    model = GigaChat()
    if not model.enabled:
        raise SystemExit("Configure GIGACHAT_CREDENTIALS before a live evaluation.")
    cases = json.loads(Path(__file__).with_name("dialogue_cases.json").read_text())
    previous = json.loads(output.read_text()) if only and output.exists() else []
    indexed = {item["id"]: item for item in previous}
    for case in cases:
        if only and case["id"] not in only:
            continue
        state = new_state()
        index = next(i for i, item in enumerate(COURSE) if item["id"] == case["case"])
        state.update(index=index, phase=case.get("phase", COURSE[index]["stage"]),
                     operation=case["step"], spoken_named=not case.get("naming", False))
        before = copy.deepcopy(state)
        reply = await respond(state, case["text"], model, history=case.get("history", []))
        progressed = any(state[key] != before[key] for key in ("phase", "index", "operation", "spoken_named", "case_correct"))
        candidate = state["pending"]
        passed = not progressed and not (candidate and candidate["key"] == "correct")
        if case["expect"] == "progress":
            passed = progressed or bool(candidate and candidate["key"] == "correct")
        if case["expect"] == "support":
            passed = passed and not candidate
        passed = passed and bool(reply) and "Не удалось проверить" not in reply
        if before["phase"] == "test":
            passed = passed and state["scores"] == {}
        result = {
            "model": model.model, **case, "passed": passed, "reply": reply, "pending": candidate,
            "after": {key: state[key] for key in ("phase", "index", "operation", "spoken_named", "case_correct", "scores")},
            "grounding": state.get("last_rag"),
        }
        if case["id"] in indexed:
            old = indexed[case["id"]]
            result["previous_attempts"] = old.get("previous_attempts", []) + [
                {key: old[key] for key in ("passed", "reply", "pending", "grounding")}
            ]
        indexed[case["id"]] = result
        results = [indexed[item["id"]] for item in cases if item["id"] in indexed]
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"{'PASS' if passed else 'FAIL'} {case['id']}", flush=True)
    failures = sum(not result["passed"] for result in results)
    print(f"{len(results) - failures}/{len(results)} cases passed", flush=True)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    raise SystemExit(bool(asyncio.run(evaluate(args.output, args.only))))
