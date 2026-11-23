"""Replay saved D01 model responses offline, including the empty nodes slots."""
import json
import hashlib
from pathlib import Path
import api_regressions as api

fixture = json.loads(Path(__file__).with_name("d01-extraction-fixture.json").read_text())
class Replay(api.NativeStub):
    async def complete(self, system, payload):
        kind = {api.RAG_PROMPT: "extract", api.VERIFIER_PROMPT: "diagnostic",
                api.SUPPORT_PLAN_PROMPT: "plan", api.SUPPORT_PROMPT: "select",
                api.GROUNDING_PROMPT: "check"}[system]
        self.calls.append(kind)
        if kind == "extract":
            assert payload["evidence_chunks"] == fixture["evidence_chunks"]
        data = {"verdict": "contradiction"} if kind == "diagnostic" else fixture["responses"][kind]
        api.validate_schema(data, api.response_schema(system, payload))
        return json.dumps(data, ensure_ascii=False)

if __name__ == "__main__":
    model = Replay({})
    api.scenario("D01_saved_empty_nodes", api.state(), fixture["text"], model,
                 lambda b, a, f, reply: a["operation"] == b["operation"] + 1
                 and a["case_correct"] == b["case_correct"] + 1 and "Резистор" in reply)
    row = api.rows[-1]
    selected = [*sorted((api.AGENT_ROOT / "app").glob("*.py")), api.AGENT_ROOT / "knowledge_base/lesson.json"]
    hashes = {str(p.relative_to(api.AGENT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    result = {"kind": "independent_offline_replay", "id": row["id"], "pass": row["pass"],
              "expected": fixture["expected"], "reply": row["reply"], "calls": model.calls,
              "before_operation": row["before"]["operation"], "after_operation": row["after"]["operation"],
              "source_hashes": hashes}
    Path(__file__).with_name("d01-replay-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
