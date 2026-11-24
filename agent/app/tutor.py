import re

from .answers import (
    allows_correct_suggestion, answer_key, branch_claims, contradicts_all_elements,
    elements_match, recall_matches, resolve_ordinal_elements,
)
from .knowledge import (
    CASES, CIRCUITS, COURSE, INTRO, STEPS, STEP_NAMES, lookup, norm, question, recall_question,
    reference_answer, METHOD_REFERENCE
)
from .rag import asks_for_help, component_help, retrieve

PHASES = {
    "trial": "Первая попытка", "method": "Способ решения", "method_compare": "Сопоставление способа",
    "guided": "Разбор с помощью", "spoken": "Проговаривание", "solo": "Самостоятельно",
    "test": "Контрольная", "done": "Результат",
}
ACTIONS = {
    "подсказка": "hint", "покажи ответ": "reference", "полный разбор": "walkthrough",
    "разговорный режим": "teacher", "режим учителя": "teacher", "краткий режим": "teacher",
    "показать варианты": "options",
}
QUESTION_START = re.compile(
    r"^(?:а )?(?:кто|где|когда|зачем|почему|сколько|"
    r"что (?:такое|значит|называется|означает)|как (?:отличить|определить|найти|понять|решить)|"
    r"в чем разница|чем отлича\w*)\b"
)
TOPIC_GUIDANCE = "Я помогаю разбирать соединение проводников."


def new_state():
    return {
        "phase": "trial", "index": 0, "operation": 0, "tries": 0, "teacher": False,
        "assisted": False, "revealed": False, "pending": None, "results": [], "errors": {},
        "case_correct": 0, "scores": {}, "reserve": None, "spoken_named": False, "show_options": False,
        "method_proposal": None, "method_comparison": None,
    }


def active_case(state):
    return CASES[state["reserve"]] if state["reserve"] else COURSE[state["index"]]


def current_question(state):
    phase = state["phase"]
    if phase in INTRO:
        return INTRO[phase]
    if phase == "done":
        return {"prompt": "Урок завершён. Можно разобрать контрольную или открыть резервную схему командой «Резерв».", "choices": []}
    case = active_case(state)
    step = "overall" if phase == "solo" else (STEPS[1:] if phase == "test" else STEPS)[state["operation"]]
    result = question(case, step)
    if phase == "spoken":
        if not state.get("spoken_named", False):
            return {**recall_question(step), "recalling": True}
        result["checking_prompt"] = result["prompt"]
        result["prompt"] = "Выполни названное действие и запиши результат."
    return result


def options_hidden(state):
    return state["phase"] == "spoken" and not state.get("show_options", False)


def pending_choice(state, current):
    pending = state["pending"]
    if isinstance(pending, dict) and pending["prompt"] == current["prompt"]:
        return next((item for item in current["choices"] if item["key"] == pending["key"]), None)
    return None


def view(state):
    phase = state["phase"]
    case = active_case(state) if phase not in ("method", "method_compare") else None
    shape = CIRCUITS[case["id"]] if case else None
    current = current_question(state)
    confirming = pending_choice(state, current) is not None
    return {
        "phase": phase, "phase_label": PHASES[phase], "case_id": case["id"] if case else None,
        "case_title": case["title"] if case else None,
        "position": state["index"] + 1 if case and phase != "done" else None,
        "total": len(COURSE), "teacher": state["teacher"], "question": current["prompt"],
        "step_label": STEP_NAMES.get(current.get("step")),
        "choices": [option["text"] for option in current["choices"]]
        if not confirming and not options_hidden(state) else [],
        "can_reveal_options": options_hidden(state) and not confirming,
        "pending": confirming, "scores": state["scores"] if phase == "done" else None,
        "circuit": {"wires": shape.wires, "boxes": shape.boxes, "junctions": shape.junctions} if shape else None,
    }


def greeting(state):
    return (
        "Привет! Я твой помощник по физике. На этом уроке научимся находить "
        "последовательные и параллельные участки на электрической схеме.\n\n"
        "Я буду задавать вопросы. Пиши ответы в чате или выбирай готовый вариант. "
        "Можно спросить, что такое узел. Если застрянешь, напиши «подсказка».\n\n"
        + current_question(state)["prompt"]
    )


def with_prompt(state, text):
    prefix = "Давай продолжим. " if state["teacher"] else ""
    return f"{text}\n\n{prefix}{current_question(state)['prompt']}"


def final_feedback(state):
    scores = "\n".join(f"- {case}: {score}/4" for case, score in state["scores"].items())
    review = sorted(state["errors"], key=state["errors"].get, reverse=True)[:2]
    coached = sum(item["assisted"] for item in state["results"])
    note = ("Повтори: " + ", ".join(STEP_NAMES.get(step, "способ решения") for step in review).lower() + "."
            if review else "Можно закрепить навык на резервной схеме.")
    return f"Контрольная завершена.\n\n{scores}\n\nСитуаций с подсказками или показом эталона: {coached}. {note}"


def start_case(state):
    state["phase"] = active_case(state)["stage"]
    state.update(operation=0, tries=0, assisted=False, revealed=False, case_correct=0,
                 spoken_named=False, show_options=False)
    return {
        "guided": "Теперь разберём схему по операциям.",
        "spoken": "Теперь тренируемся проговаривать способ. Сначала назови действие, затем выполни его.",
        "solo": "Самостоятельное решение. Опиши вид соединения и обоснуй его. Можешь записать, между какими узлами лежат ветви.",
        "test": "Контрольная. Ответь на четыре вопроса по схеме. Обратная связь и разбор будут после сдачи обеих схем.",
    }[state["phase"]]


def advance_case(state):
    old_phase = state["phase"]
    state["results"].append({
        "case_id": active_case(state)["id"], "assisted": state["assisted"],
        "revealed": state["revealed"], "correct": state["case_correct"],
    })
    if state["phase"] == "test":
        state["scores"][active_case(state)["id"]] = state["case_correct"]
    if state["index"] + 1 == len(COURSE):
        state["phase"] = "done"
        return final_feedback(state)
    state["index"] += 1
    intro = start_case(state)
    return with_prompt(state, intro if state["phase"] != old_phase else "Следующая схема.")


def grade(state, selected):
    current = current_question(state)
    correct = selected["key"] == "correct"
    phase = state["phase"]
    if phase == "solo" and not correct:
        state["assisted"] = True

    if phase == "test":
        state["case_correct"] += int(correct)
        if not correct:
            step = current["step"]
            state["errors"][step] = state["errors"].get(step, 0) + 1
        state["operation"] += 1
        if state["operation"] == 4:
            return advance_case(state)
        return with_prompt(state, "Ответ записан.")

    if not correct:
        step = current.get("step", phase)
        state["tries"] += 1
        state["errors"][step] = state["errors"].get(step, 0) + 1
        if "hints" in current:
            state["assisted"] = True
        if phase == "solo":
            return with_prompt(state, "В этом выводе есть ошибка или ему не хватает основания. " + current["hints"][0])
        feedback = selected["feedback"]
        if state["tries"] >= 2 and "hints" in current:
            feedback = current["hints"][-1]
        return with_prompt(state, feedback)

    state["tries"] = 0
    if phase == "method":
        return with_prompt(state, start_case(state))
    if phase == "spoken":
        state["show_options"] = False
        if not state.get("spoken_named", False):
            state["spoken_named"] = True
            return with_prompt(state, "Действие подходит.")
        state["spoken_named"] = False
    state["case_correct"] += int(not state["revealed"])
    state["operation"] += 1
    if phase == "solo" or state["operation"] == len(STEPS):
        result = advance_case(state)
        acknowledgement = "Разбор завершён." if phase in ("guided", "spoken") else "Самостоятельный ответ принят."
        return acknowledgement + "\n\n" + result
    return with_prompt(state, "Верно. Этот шаг выполнен.")


def help_reply(state, text):
    request = norm(text)
    action = ACTIONS.get(request)
    target = re.fullmatch(r"(покажи ответ|полный разбор) c?\s*(\d{1,2})", request)
    reserve = re.fullmatch(r"резерв(?: c?\s*(11|12))?", request)
    if reserve and state["phase"] == "done":
        state["reserve"] = f"C{reserve[1]}" if reserve[1] else ("C12" if state["reserve"] == "C11" else "C11")
        return "Резервная схема открыта. Попробуй самостоятельно объяснить связи, затем запроси «Полный разбор»."
    if action == "teacher":
        state["teacher"] = request != "краткий режим"
        return with_prompt(state, "Объясняю в разговорном стиле." if state["teacher"] else "Перехожу к кратким формулировкам.")
    if action == "options" and state["phase"] == "spoken":
        state.update(assisted=True, show_options=True)
        return with_prompt(state, "Варианты открыты.")
    is_reference = bool(target or action in ("reference", "walkthrough"))
    is_lookup = request == "справка" or request.startswith("справка ") or QUESTION_START.match(request)
    if state["phase"] == "test" and (is_reference or is_lookup or action == "hint"):
        return with_prompt(state, "Разбор и справка доступны после сдачи контрольной.")
    if is_reference:
        case = CASES.get(f"C{int(target[2]):02}") if target else active_case(state)
        if not case:
            return "Такой схемы нет. Укажи номер от 1 до 12."
        if case["stage"] == "test" and state["phase"] != "done":
            return "Эта схема открывается после сдачи контрольной."
        if case["id"] == active_case(state)["id"] and state["phase"] in ("trial", "guided", "spoken", "solo"):
            state.update(assisted=True, revealed=True)
        full = (target[1] == "полный разбор") if target else action == "walkthrough"
        return reference_answer(case, full=full)
    if is_lookup:
        matches = lookup(text)
        if not matches:
            return with_prompt(state, TOPIC_GUIDANCE + " Можешь спросить об узлах, последовательном или параллельном соединении.")
        return "\n\n".join(f"**{item['title']}.** {item['text']}" for item in matches)
    if action == "hint":
        hints = current_question(state).get("hints")
        if not hints:
            return with_prompt(state, "Попробуй выделить элементы и проследить их выводы.")
        state["assisted"] = True
        state["tries"] += 1
        return with_prompt(state, hints[min(state["tries"] - 1, len(hints) - 1)])
    return None


def support_reply(state, text, result=None):
    current = current_question(state)
    intent = result.intent if result else "unsure"
    if intent == "off_topic":
        return with_prompt(state, TOPIC_GUIDANCE)
    if current.get("recalling") and intent in ("partial", "clarify", "hint", "unsure"):
        return help_reply(state, "подсказка")
    if intent in ("hint", "unsure"):
        state["assisted"] = True
    if result and result.explanation:
        return result.explanation
    if result and result.quotes:
        return "\n\n".join(quote.text for quote in result.quotes)
    if intent == "partial":
        return with_prompt(state, "Ответ пока неполный. Допиши недостающие элементы или связи.")
    if intent == "clarify":
        return with_prompt(state, "Уточни, к какому элементу или участку относится твой ответ.")
    if current.get("step") == "nodes":
        explanation = component_help(active_case(state), text)
        if explanation:
            return with_prompt(state, "Давай разберём связь. " + explanation["text"])
    return help_reply(state, "подсказка")


def add_explanation(reply, result):
    if not result or not result.question_text:
        return reply
    explanation = result.explanation or "\n\n".join(quote.text for quote in result.quotes)
    if not explanation:
        return reply
    acknowledgement, separator, following = reply.partition("\n\n")
    return "\n\n".join(part for part in (acknowledgement, explanation, following) if part)


def verified_training_answer(state, text):
    current = current_question(state)
    if ("?" in text or QUESTION_START.match(norm(text)) or asks_for_help(text)
            or state["phase"] in ("test", "done") or current.get("recalling")):
        return False
    shape = CIRCUITS[active_case(state)["id"]]
    if state["phase"] == "trial":
        branches, _ = shape.parallel[0]
        expected = {
            key: set(re.findall(r"[rl]\d+", branch.lower()))
            for key, branch in zip(("upper", "lower"), branches)
        }
        return branch_claims(text, strict=True) == expected
    if current.get("step") in ("elements", "nodes"):
        return allows_correct_suggestion(
            state["phase"], current, resolve_ordinal_elements(text, shape.components),
        )
    return False


async def respond(state, text, model, *, history=()):
    text = text.strip()
    request = norm(text)
    if state["phase"] in ("method", "method_compare"):
        # This is a discussion of a finite procedure, not another physical grade.
        if request in ("подсказка", "покажи ответ", "полный разбор"):
            state["assisted"] = True
            return with_prompt(state, "Образец способа решения:\n" + METHOD_REFERENCE)
        help_text = help_reply(state, text)
        if help_text is not None:
            return help_text
        if not text:
            return with_prompt(state, "Напиши свой план." if state["phase"] == "method" else "Напиши результат сопоставления.")
        if state["phase"] == "method":
            unsure = asks_for_help(text)
            relevant = re.search(r"элемент\w*|резистор\w*|ламп\w*|уз(?:ел|л)\w*|вывод\w*|последоват\w*|параллел\w*|соедин\w*", request)
            if not unsure and not relevant:
                return with_prompt(state, "Опиши действия со схемой: например, какие обозначения и подключения будешь искать.")
            state.update(phase="method_compare", method_proposal=text, pending=None)
            state["assisted"] = state["assisted"] or unsure
            lead = "Разберём порядок по образцу." if unsure else "Твой план записан. Вот образец для сопоставления."
            return with_prompt(state, lead + "\n\n" + METHOD_REFERENCE)
        adopting = request in ("план совпадает", "использовать образец")
        comparing = re.search(r"добав\w*|уточн\w*|исправ\w*|поменя\w*|пропуст\w*|совпад\w*|сначала|затем|потом", request)
        if not adopting and not comparing:
            return with_prompt(state, "Укажи изменение в плане или напиши «План совпадает» либо «Использовать образец».")
        state["method_comparison"] = text
        state["method_completed"] = True
        return with_prompt(state, "Способ зафиксирован. " + start_case(state))
    if state["pending"] is not None:
        selected = pending_choice(state, current_question(state))
        if selected and request in ("подтвердить", "да"):
            state["pending"] = None
            return grade(state, selected)
        if request in ("подтвердить", "да"):
            return with_prompt(state, "Запиши ответ ещё раз для проверки.")
        if request in ("исправить", "нет"):
            state["pending"] = None
            return with_prompt(state, "Напиши ответ точнее или выбери подходящий вариант.")
        if request.startswith(("нет ", "исправить ")):
            # The leading rejection is an explicit reply to the confirmation
            # control, even when a question follows it in the same turn.
            state["pending"] = None
    if request in ("подтвердить", "да"):
        return with_prompt(state, "Сейчас нет ответа, ожидающего подтверждения. Запиши своё решение.")
    current = current_question(state)
    verified_answer = verified_training_answer(state, text)
    exact_choice = any(answer_key(item["text"], current.get("step")) == answer_key(text, current.get("step"))
                       for item in current["choices"])
    command = request in ACTIONS and request != "подсказка" or bool(re.fullmatch(
        r"(?:покажи ответ|полный разбор|резерв)(?: c?\s*\d{1,2})?", request
    ))
    rag_result = None
    rag_status = "disabled"
    if text and not exact_choice and not command and state["phase"] not in ("test", "done") and hasattr(model, "rag"):
        sources = retrieve(active_case(state), current, text, history=history)
        task = {**current, "pending_answer": state["pending"], "case_id": active_case(state)["id"]}
        rag_result, rag_status = await model.rag(text, task, sources, history=history)
        if rag_result and rag_result.withdraws_previous:
            state["pending"] = None
        if getattr(model, "enabled", False):
            state["last_rag"] = {
                "status": rag_status, "source_ids": rag_result.source_ids if rag_result else [],
                "retrieved_source_ids": [source["id"] for source in sources],
                "intent": rag_result.intent if rag_result else None,
                "answer_verified": rag_result.answer_verified if rag_result else False,
                "help_scope": rag_result.help_scope if rag_result else None,
                "assessment": rag_result.assessment_status if rag_result else None,
            }
        verified_model_answer = bool(rag_result and rag_result.answer_verified)
        if rag_status == "unavailable" and not verified_model_answer and not verified_answer:
            return "Помощник временно недоступен. Ответ не оценён, прогресс сохранён.\n\n" + support_reply(state, text, rag_result)
        if rag_result and not verified_model_answer and not verified_answer:
            return support_reply(state, text, rag_result)
        if asks_for_help(text) and not verified_model_answer and not verified_answer:
            return support_reply(state, text)
    if (getattr(model, "enabled", False) and hasattr(model, "rag")
            and rag_status in ("unavailable", "unclear") and not verified_answer):
        return support_reply(state, text)
    model_verified = bool(rag_result and rag_result.answer_verified)
    if asks_for_help(text) and not model_verified and not verified_answer and state["phase"] not in ("test", "done"):
        return support_reply(state, text)
    help_text = None if model_verified else help_reply(state, text)
    if help_text is not None:
        return help_text
    phase = state["phase"]
    if phase == "done":
        return "Урок завершён. Доступны справка, эталонные ответы и резервные схемы."
    if not text:
        return with_prompt(state, "Напиши ответ на текущий вопрос.")
    if phase == "trial":
        case = active_case(state)
        branches, _ = CIRCUITS[case["id"]].parallel[0]
        upper, lower = (branch.strip("()") for branch in branches)
        expected = {
            key: set(re.findall(r"[rl]\d+", branch.lower()))
            for key, branch in zip(("upper", "lower"), branches)
        }
        unsure = bool(re.search(r"не знаю|откуда (?:я |мне )?зна|затрудняюсь|не получается", request))
        claims = branch_claims(text, strict=True)
        explicit_claims = claims
        if rag_result and rag_result.answer_verified:
            if rag_result.branches:
                claims = {}
            for branch in rag_result.branches:
                claims.setdefault(branch.position, set()).update(element.lower() for element in branch.elements)
            mentioned = set(re.findall(r"\b[rl][1-9]\d*\b", resolve_ordinal_elements(text, CIRCUITS[case["id"]].components)))
            claimed = set().union(*claims.values())
            if claimed != mentioned or claims.get("upper", set()) & claims.get("lower", set()):
                claims = {}
            if re.search(r"\b(?:не|или)\b", request):
                claims = {}
            if explicit_claims and explicit_claims != claims:
                claims = explicit_claims
            unsure = False
        if claims != expected and not unsure:
            if claims:
                if all(values <= expected.get(position, set()) for position, values in claims.items()):
                    return support_reply(state, text, rag_result) if rag_result and (
                        rag_result.explanation or rag_result.quotes
                    ) else with_prompt(state, "Ответ пока неполный. Укажи расположение остальных элементов.")
                return with_prompt(state, "В расположении резисторов есть ошибка. Проследи каждую ветвь от A до B и проверь подписи.")
            return with_prompt(state, "Напиши отдельно, какие обозначения стоят сверху и какие снизу. Если трудно, напиши «не знаю».")
        state.update(phase="method", operation=0, tries=0, case_correct=int(not unsure), pending=None,
                     assisted=state["assisted"] or unsure)
        feedback = "Верно." if not unsure else "Давай разберём вместе."
        return add_explanation(with_prompt(
            state,
            f"{feedback} В верхней ветви стоят {upper}, в нижней — {lower}. "
            "Перед разбором составим способ решения.",
        ), rag_result)

    current = current_question(state)
    state["pending"] = None
    choices = current["choices"]
    step = current.get("step")
    correct = next((choice for choice in choices if choice["key"] == "correct"), None)
    naming_action = phase == "spoken" and not state.get("spoken_named", False)
    if naming_action:
        if rag_result and rag_result.answer_verified and rag_result.option:
            selected = choices["ABCD".index(rag_result.option)]
            return add_explanation(grade(state, selected), rag_result)
        if recall_matches(step, text):
            response = grade(state, correct)
            _, separator, result = text.partition(".")
            if step == "elements" and separator:
                execution = current_question(state)
                if elements_match(result, execution["reference"]):
                    selected = next(item for item in execution["choices"] if item["key"] == "correct")
                    return grade(state, selected)
            return response
        if options_hidden(state):
            return with_prompt(state, "Сначала назови само действие. Если застрял, попроси показать варианты.")
    numbered = re.fullmatch(r"(?:вариант )?(\d{1,2})", request)
    if numbered:
        if options_hidden(state):
            return with_prompt(state, "Запиши результат своими словами или попроси показать варианты.")
        index = int(numbered[1]) - 1
        if 0 <= index < len(choices):
            return grade(state, choices[index])
        return with_prompt(state, f"Выбери номер от 1 до {len(choices)}.")
    normalized = answer_key(text, step)
    exact = next((item for item in choices if answer_key(item["text"], step) == normalized), None)
    if exact:
        return grade(state, exact)
    if naming_action:
        return with_prompt(state, "Назови действие своими словами или выбери его в списке.")
    if phase == "test":
        return with_prompt(state, "На контрольной выбери один вариант и нажми «Отправить вариант».")
    if rag_result and rag_result.answer_verified and rag_result.option:
        selected = choices["ABCD".index(rag_result.option)]
        if selected["key"] == "correct" and contradicts_all_elements(
            text, step, CIRCUITS[active_case(state)["id"]],
        ):
            return support_reply(state, text)
        return add_explanation(grade(state, selected), rag_result)
    if step == "elements" and elements_match(text, current["reference"]):
        return grade(state, correct)
    if step == "nodes" and allows_correct_suggestion(
        phase, current, resolve_ordinal_elements(text, CIRCUITS[active_case(state)["id"]].components),
    ):
        return grade(state, correct)
    grading_prompt = question(active_case(state), step)["prompt"] if phase == "spoken" else current["prompt"]
    if rag_result:
        selected = "ABCD".index(rag_result.option) if rag_result.option else None
    elif hasattr(model, "rag"):
        selected = None
    else:
        selected, _ = await model.interpret(text, grading_prompt, choices)
    if selected is not None and 0 <= selected < len(choices):
        interpreted = choices[selected]
        if interpreted["key"] == "correct" and contradicts_all_elements(
            text, step, CIRCUITS[active_case(state)["id"]],
        ):
            return help_reply(state, "подсказка")
        if rag_result or interpreted["key"] != "correct" or allows_correct_suggestion(phase, current, text):
            state["pending"] = {"key": interpreted["key"], "prompt": current["prompt"]}
            if rag_result and rag_result.needs_clarification:
                state["assisted"] = True
                return f"Уточню: ты имеешь в виду «{interpreted['text']}»? Подтверди или исправь."
            return f"Я понял твой ответ так: «{interpreted['text']}». Подтверди или исправь формулировку."
    if options_hidden(state):
        return with_prompt(state, "Не удалось проверить формулировку. Уточни результат или попроси показать варианты.")
    if rag_result and rag_result.intent == "answer":
        return with_prompt(state, "Уточни ответ: назови элементы и объясни, по какому признаку ты сделал вывод.")
    return with_prompt(state, TOPIC_GUIDANCE + " Можешь задать вопрос по теме, ответить своими словами или выбрать вариант.")
