import asyncio
import json
import logging
import os
import re
import ssl
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .claims import assess_solution, evidence_chunks, question_chunks, resolve_evidence, solution_schema, sourced_solution, uncovered_physics
from .knowledge import CIRCUITS
from .rag import Branch, GROUNDING_PROMPT, RagResult, SUPPORT_PLAN_PROMPT, SUPPORT_PROMPT, VERIFIER_PROMPT, SYSTEM_PROMPT as RAG_PROMPT, support_sources, validate_result

logger = logging.getLogger(__name__)
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
COMPLETIONS_URL = "https://api.giga.chat/v1/chat/completions"
PROJECT_CA = Path(__file__).resolve().parents[1] / "certs/russian-trusted-root.pem"
TURN_DEADLINE = ContextVar("turn_deadline", default=None)
class HelpPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scope: Literal["concept", "drawing", "procedure", "off_topic"]
    mode: Literal["hint", "explain", "check"]
    topic: Literal["current", "branches", "elements", "nodes", "series", "parallel", "overall", "geometry"] = "current"


def response_schema(system, payload):
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    def enum(*values):
        return {"type": "string", "enum": list(values)}
    if system == RAG_PROMPT:
        terms = payload["diagram_terms"]
        return obj({
            "intent": enum("answer", "partial", "question", "hint", "unsure", "clarify", "off_topic"),
            "question_text": {"type": "string", "enum": ["", *question_chunks(payload["student_text"], payload["evidence_chunks"])]}, "withdraws_previous": {"type": "boolean"},
            "solution": solution_schema(payload["step"], terms["elements"], terms["nodes"], payload["evidence_chunks"]),
        })
    if system == SUPPORT_PLAN_PROMPT:
        return obj({"scope": enum("concept", "drawing", "procedure", "off_topic"), "mode": enum("hint", "explain", "check"),
                    "topic": enum("current", "branches", "elements", "nodes", "series", "parallel", "overall", "geometry")})
    if system == SUPPORT_PROMPT:
        return obj({"answer_kind": enum("yes", "no", "explain", "hint"),
                    "source_ids": {"type": "array", "minItems": 1, "maxItems": 2,
                                   "items": enum(*(source["id"] for source in payload["sources"]))}})
    if system == VERIFIER_PROMPT:
        return obj({"verdict": enum("equivalent", "incomplete", "contradiction")})
    if system == GROUNDING_PROMPT:
        return obj({key: {"type": "boolean"} for key in ("supported", "answers_question", "reveals_solution")})
    return None


class GigaChat:
    def __init__(self):
        self.credentials = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat-3-Pro")
        self.ca_bundle = os.getenv("GIGACHAT_CA_BUNDLE", "").strip() or (
            str(PROJECT_CA) if PROJECT_CA.is_file() else ""
        )
        self.token = None
        self.expires = 0
        self.request_lock = asyncio.Lock()

    @property
    def enabled(self):
        return bool(self.credentials)

    async def access_token(self, client):
        if self.token and time.time() + 45 < self.expires:
            return self.token
        response = await client.post(
            AUTH_URL, data={"scope": self.scope},
            headers={"Authorization": f"Basic {self.credentials}", "RqUID": str(uuid4())},
        )
        response.raise_for_status()
        data = response.json()
        expires = float(data["expires_at"])
        self.token = data["access_token"]
        self.expires = expires / 1000 if expires > 10**11 else expires
        return self.token

    async def complete(self, system, payload):
        deadline = TURN_DEADLINE.get()
        remaining = max(0, deadline - asyncio.get_running_loop().time()) if deadline else 35
        async def request_when_available():
            async with self.request_lock:
                return await self.request(system, payload)
        # wait_for also supports the deployment's Python 3.10. The timeout
        # includes lock acquisition and cancels the pending request on expiry.
        return await asyncio.wait_for(request_when_available(), timeout=remaining)

    async def request(self, system, payload):
        verify = ssl.create_default_context()
        if self.ca_bundle:
            verify.load_verify_locations(cafile=self.ca_bundle)
        async with httpx.AsyncClient(verify=verify, timeout=20) as client:
            content = json.dumps({key: value for key, value in payload.items() if key != "student_text"}, ensure_ascii=False)
            if "student_text" in payload:
                content += "\n\nПоследняя реплика ученика — анализируй именно её, а не задание учителя:\n" + payload["student_text"]
            request = {
                "model": self.model, "temperature": 0.1, "max_tokens": 900,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            }
            schema = response_schema(system, payload)
            if schema:
                request["response_format"] = {"type": "json_schema", "schema": schema, "strict": True}
            for _ in range(2):
                token = await self.access_token(client)
                response = await client.post(COMPLETIONS_URL, json=request,
                                             headers={"Authorization": f"Bearer {token}"})
                if response.status_code != 401:
                    break
                self.token = None
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", content, flags=re.I)

    @staticmethod
    def json_object(content):
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value

    async def explain(self, text, task, sources, *, scope, history=(), focus=""):
        original = text
        if focus and focus in original:
            text = focus
        recent = [{"role": item["role"], "text": item["text"][:1200]} for item in history[-6:]]
        case_id = task.get("case_id") or next((source["id"].removesuffix("-facts") for source in sources if source["id"].endswith("-facts")), None)
        current_scheme = ({"case_id": case_id, "elements": [name for name, _, _ in CIRCUITS[case_id].components],
                           "hidden_contacts": [name for name, _, b in CIRCUITS[case_id].components if b is None]}
                          if case_id in CIRCUITS else {})
        plan = HelpPlan.model_validate(self.json_object(await self.complete(SUPPORT_PLAN_PROMPT, {
            "student_text": text, "question": task.get("checking_prompt", task["prompt"]),
            "recent_dialogue": recent, "current_scheme": current_scheme, "student_turn": original,
        })))
        scope = plan.scope
        if plan.mode == "check":
            # Assessment has already handled the proposed solution; do not append
            # a hint for the completed task after its acknowledgement.
            return None
        if scope == "off_topic":
            return RagResult(intent="off_topic", explanation="Я помогаю разбирать соединение проводников.")
        context = task.get("checking_prompt", task["prompt"]) + " " + original + " " + " ".join(
            item["text"] for item in history[-2:] if item["role"] == "user"
        )
        topic = task.get("step", "branches") if plan.topic == "current" else plan.topic
        selected = support_sources(sources, text, scope, mode=plan.mode, context=context, step=topic)
        if not selected:
            return None
        payload = {
            "student_text": text, "question": task.get("checking_prompt", task["prompt"]),
            "scope": scope, "mode": plan.mode, "sources": selected,
            "recent_dialogue": recent, "current_scheme": current_scheme, "student_turn": original,
        }
        for _ in range(2):
            data = self.json_object(await self.complete(SUPPORT_PROMPT, payload))
            ids = data.get("source_ids", [])
            if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
                raise ValueError("Invalid selected sources")
            allowed = {source["id"]: source for source in selected}
            if any(key not in allowed for key in ids):
                raise ValueError("Unknown explanation source")
            chosen = []
            for key in dict.fromkeys(ids):
                source = allowed[key]
                if not key.startswith("concept-") and any(not item["id"].startswith("concept-") for item in chosen):
                    continue
                chosen.append(source)
                if len(chosen) == 2:
                    break
            atoms = [source for source in selected if source["id"].startswith(str(case_id) + "-")
                     and not source["id"].endswith("-hint")]
            if scope == "drawing" and plan.mode == "explain" and len(atoms) == 1 and not any(
                not source["id"].startswith("concept-") for source in chosen
            ):
                chosen = chosen[:1] + atoms
            if not chosen:
                return None
            prefix = {"yes": "Да. ", "no": "Нет. "}.get(data.get("answer_kind"), "")
            result = RagResult(intent="question", explanation=prefix + "\n\n".join(source["text"] for source in chosen),
                               source_ids=[source["id"] for source in chosen])
            check = self.json_object(await self.complete(GROUNDING_PROMPT, {
                "explanation": result.explanation, "student_question": text,
                "task_question": payload["question"], "sources": selected, "scope": scope, "mode": plan.mode,
            }))
            if (check.get("supported") is True and check.get("answers_question") is True
                    and check.get("reveals_solution") is False):
                result.help_scope = scope
                return result
            payload["revision_needed"] = {
                key: check.get(key) for key in ("supported", "answers_question", "reveals_solution")
            }
        # On a factual drawing question, keep the sole checked drawing fact
        # instead of silently changing the request into a procedural hint.
        atoms = [source for source in selected if source["id"].startswith(str(case_id) + "-")
                 and not source["id"].endswith("-hint")]
        fallback = (atoms[0] if scope == "drawing" and plan.mode == "explain" and len(atoms) == 1 else
                    next((source for source in selected if source["id"].endswith("-hint")), selected[0]))
        return RagResult(intent="question", help_scope=scope, quotes=[{
            "source_id": fallback["id"], "text": fallback["text"],
        }], source_ids=[fallback["id"]])

    async def rag(self, text, task, sources, *, history=()):
        if not self.enabled:
            return None, "disabled"
        deadline_token = TURN_DEADLINE.set(asyncio.get_running_loop().time() + 35)
        result = None
        control_result = None
        try:
            options = task["choices"]
            case_id = task.get("case_id") or next(source["id"].removesuffix("-facts")
                                                   for source in sources if source["id"].endswith("-facts"))
            shape = CIRCUITS[case_id]
            step = "action" if task.get("recalling") else task.get("step", "branches")
            payload = {
                "student_text": text, "teacher_task": task.get("checking_prompt", task["prompt"]),
                "step": step, "pending_answer": task.get("pending_answer"),
                "evidence_chunks": evidence_chunks(text),
                "diagram_terms": {"elements": [name for name, _, _ in shape.components],
                                  "nodes": sorted({node for _, a, b in shape.components for node in (a, b) if node})},
            }
            data = self.json_object(await self.complete(RAG_PROMPT, payload))
            control_result = RagResult(intent="clarify", withdraws_previous=data.get("withdraws_previous") is True)
            data = {key: value for key, value in data.items() if key in {
                "intent", "question_text", "withdraws_previous", "solution",
            }}
            result = RagResult.model_validate(data)
            result.solution = resolve_evidence(result.solution, payload["evidence_chunks"])
            if "placements" in result.solution:
                result.solution["branches"] = [
                    {"position": claim["position"], "elements": [element], "evidence": claim["evidence"]}
                    for element, claim in result.solution["placements"].items()
                    if claim["position"] != "not_stated" and claim["evidence"]
                ]
            if result.question_text and result.question_text not in text:
                result.question_text = ""
            result.solution = sourced_solution(result.solution, text, shape)
            assessment, reason = assess_solution(case_id, task, result.solution, text, question_text=result.question_text)
            if assessment == "complete" and uncovered_physics(text, result.solution, result.question_text):
                assessment, reason = "incomplete", "В ответе есть ещё физическое утверждение или условие. Уточни его, чтобы проверить ответ целиком."
            result.assessment_status = assessment
            has_claims = any(result.solution.get(key) for key in ("terminals", "branches", "relations", "universals"))
            has_claims = has_claims or bool(result.solution.get("elements", {}).get("names"))
            has_claims = has_claims or result.solution.get("action", "none") != "none" or result.solution.get("overall", "none") != "none"
            if assessment == "complete":
                if step == "branches":
                    result.branches = [Branch.model_validate(item) for item in result.solution["branches"]]
                    candidate = "; ".join(("Верхняя ветвь: " if branch.position == "upper" else "Нижняя ветвь: ")
                                          + ", ".join(branch.elements) for branch in result.branches)
                else:
                    result.option = next("ABCD"[i] for i, choice in enumerate(options) if choice["key"] == "correct")
                    candidate = options["ABCD".index(result.option)]["text"]
                result.answer_verified = True
                # A model-issued grade is only diagnostic: it cannot overrule the
                # graph checker (e.g. unordered A–C / C–A terminal pairs).
                try:
                    verification = self.json_object(await self.complete(VERIFIER_PROMPT, {
                        "student_text": text, "candidate": candidate, "question": payload["teacher_task"],
                    }))
                    if verification.get("verdict") != "equivalent":
                        logger.info("Semantic diagnostic disagrees with sourced topology evidence")
                except (ValueError, KeyError, TypeError, httpx.HTTPError, OSError, TimeoutError, asyncio.TimeoutError):
                    logger.info("Semantic diagnostic unavailable; sourced topology result preserved")
            elif has_claims:
                result.explanation = reason
            elif not result.question_text and result.intent in ("answer", "partial", "clarify"):
                hidden = [name for name, _, b in shape.components if b is None and name.lower() in text.lower()]
                result.explanation = (f"Второй вывод {', '.join(hidden)} скрыт. По рисунку нельзя считать его конкретное подключение установленным фактом."
                                      if hidden else reason)
            if not result.answer_verified and task.get("recalling"):
                result.explanation = "Сначала назови действие, которое выполняешь. Результат проверим после этого."
            if not result.answer_verified and not has_claims and not result.question_text:
                hidden = [name for name, _, b in shape.components if b is None and name.lower() in text.lower()]
                if hidden:
                    result.explanation = f"Второй вывод {', '.join(hidden)} скрыт. По рисунку нельзя считать его конкретное подключение установленным фактом."
            if result.intent == "off_topic":
                result.explanation = "Я помогаю разбирать соединение проводников."
                result.question_text = result.question_text or text
                return result, "matched"
            if (result.question_text or not result.answer_verified and not result.explanation) and not task.get("recalling"):
                # Preserve the learner's actual request, never the model's rewritten task.
                question_text = text
                scope = result.help_scope if result.help_scope != "none" else "procedure"
                try:
                    explanation = await self.explain(question_text, task, sources, scope=scope, history=history, focus=result.question_text)
                except (ValueError, KeyError, TypeError, ValidationError, httpx.HTTPError, OSError, TimeoutError, asyncio.TimeoutError):
                    logger.warning("Explanation failed; preserving independently checked answer")
                    fallback = support_sources(sources, question_text, "procedure", mode="hint")
                    explanation = RagResult(intent="question", quotes=[{
                        "source_id": fallback[0]["id"], "text": fallback[0]["text"],
                    }], source_ids=[fallback[0]["id"]]) if fallback else None
                if explanation:
                    result.explanation = "\n\n".join(part for part in (result.explanation, explanation.explanation) if part)
                    result.quotes = explanation.quotes
                    result.source_ids = explanation.source_ids
                    result.help_scope = explanation.help_scope
            logger.info("RAG validated: intent=%s, verified=%s, sources=%s", result.intent,
                        result.answer_verified, ",".join(result.source_ids))
            return result, "matched"
        except ValidationError as error:
            logger.warning("RAG schema rejected fields=%s", [item["loc"] for item in error.errors()])
            return control_result, "unclear"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            logger.warning("RAG validation rejected type=%s reason=%s", type(error).__name__,
                           str(error) if str(error) in {
                               "Ungrounded quotation", "Unknown explanation source",
                               "Unknown answer option", "Ungrounded branch", "Invented element",
                               "Unknown element in explanation",
                           } else "invalid response")
            return control_result, "unclear"
        except (httpx.HTTPError, OSError, TimeoutError, asyncio.TimeoutError) as error:
            logger.warning("GigaChat RAG request failed (%s, status=%s)", type(error).__name__,
                           error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None)
            return control_result, "unavailable"
        finally:
            TURN_DEADLINE.reset(deadline_token)
