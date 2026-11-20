import re

from .answers import allows_correct_suggestion, answer_key, elements_match, recall_matches, try_answer
from .knowledge import (
    CASES, CIRCUITS, COURSE, INTRO, STEPS, STEP_NAMES, lookup, norm, question, recall_question,
    reference_answer
)

PHASES = {
    "goal": "Цель", "trial": "Первая попытка", "compare": "Сравнение", "method": "Способ решения",
    "guided": "Разбор с помощью", "spoken": "Проговаривание", "solo": "Самостоятельно",
    "test": "Контрольная", "done": "Результат",
}
ACTIONS = {
    "подсказка": "hint", "покажи ответ": "reference", "полный разбор": "walkthrough",
    "режим учителя": "teacher", "краткий режим": "teacher", "показать варианты": "options",
}


def new_state():
    return {
        "phase": "goal", "index": 0, "operation": 0, "tries": 0, "teacher": False,
        "assisted": False, "revealed": False, "pending": None, "results": [], "errors": {},
        "case_correct": 0, "scores": {}, "reserve": None, "spoken_named": False, "show_options": False,
    }


def active_case(state):
    return CASES[state["reserve"]] if state["reserve"] else COURSE[state["index"]]


def current_question(state):
    phase = state["phase"]
    if phase in INTRO:
        return INTRO[phase]
    if phase == "trial":
        return {"prompt": "Попробуй описать соединение элементов на схеме. Здесь важна первая попытка.", "choices": []}
    if phase == "done":
        return {"prompt": "Урок завершён. Можно разобрать контрольную или открыть резервную схему командой «Резерв».", "choices": []}
    case = active_case(state)
    step = "overall" if phase == "solo" else (STEPS[1:] if phase == "test" else STEPS)[state["operation"]]
    result = question(case, step)
    if phase == "spoken":
        if not state.get("spoken_named", False):
            return recall_question(step)
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
    case = active_case(state) if phase not in ("goal", "method") else None
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
        "Привет! Научимся определять соединение проводников по схеме. "
        "В последовательной цепочке между элементами нет ответвлений. "
        "У параллельных ветвей совпадают оба узла подключения.\n\n"
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
    if phase == "compare":
        state["phase"] = "method"
        note = {
            "same": "Вывод совпал. Теперь сформулируй способ, которым можно его обосновать.",
            "different": "Разберём расхождение: один общий узел ещё не делает ветви параллельными.",
            "unclear": "Начни с двух выводов каждого элемента. Дальше проверим способ по шагам.",
        }[selected["key"]]
        return with_prompt(state, note)
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
    if phase == "goal":
        state["phase"] = "trial"
        return with_prompt(state, "Потренируемся распознавать способ соединения.")
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
        state["teacher"] = request == "режим учителя"
        return with_prompt(state, "Объясняю в разговорном стиле." if state["teacher"] else "Перехожу к кратким формулировкам.")
    if action == "options" and state["phase"] == "spoken":
        state.update(assisted=True, show_options=True)
        return with_prompt(state, "Варианты открыты.")
    is_reference = bool(target or action in ("reference", "walkthrough"))
    is_lookup = request.startswith("справка") or (
        text.strip().endswith("?") and bool(re.match(r"(что такое|что значит|как отличить) ", request))
    )
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
            return "В базе урока есть материалы об узлах, последовательном, параллельном и смешанном соединениях."
        return "\n\n".join(f"**{item['title']}.** {item['text']}" for item in matches)
    if action == "hint":
        hints = current_question(state).get("hints")
        if not hints:
            return with_prompt(state, "Попробуй выделить элементы и проследить их выводы.")
        state["assisted"] = True
        state["tries"] += 1
        return with_prompt(state, hints[min(state["tries"] - 1, len(hints) - 1)])
    return None


async def respond(state, text, model):
    text = text.strip()
    request = norm(text)
    if state["pending"] is not None:
        selected = pending_choice(state, current_question(state))
        state["pending"] = None
        if selected and request in ("подтвердить", "да"):
            return grade(state, selected)
        if request in ("подтвердить", "да"):
            return with_prompt(state, "Запиши ответ ещё раз для проверки.")
        if request in ("исправить", "нет"):
            return with_prompt(state, "Напиши ответ точнее или выбери подходящий вариант.")
    help_text = help_reply(state, text)
    if help_text is not None:
        return help_text
    phase = state["phase"]
    if phase == "done":
        return "Урок завершён. Доступны справка, эталонные ответы и резервные схемы."
    if not text:
        return with_prompt(state, "Напиши, что ты видишь на схеме, или выбери вариант ответа.")
    if phase == "trial":
        if not try_answer(text):
            return with_prompt(state, "Опиши элементы или связи на схеме. Если пока не получается, так и напиши.")
        state["phase"] = "compare"
        return with_prompt(state, "Ответ учителя:\n\n" + reference_answer(active_case(state)))

    current = current_question(state)
    choices = current["choices"]
    step = current.get("step")
    correct = next((choice for choice in choices if choice["key"] == "correct"), None)
    naming_action = phase == "spoken" and not state.get("spoken_named", False)
    if naming_action:
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
    if phase == "compare":
        return with_prompt(state, "Выбери, совпал ли твой вывод с ответом учителя.")
    if phase == "test":
        return with_prompt(state, "На контрольной выбери один вариант и нажми «Ответить».")
    if step == "elements" and elements_match(text, current["reference"]):
        return grade(state, correct)
    if step == "nodes" and allows_correct_suggestion(phase, current, text):
        return grade(state, correct)
    grading_prompt = question(active_case(state), step)["prompt"] if phase == "spoken" else current["prompt"]
    selected, status = await model.interpret(text, grading_prompt, choices)
    if selected is not None and 0 <= selected < len(choices):
        interpreted = choices[selected]
        if interpreted["key"] != "correct" or allows_correct_suggestion(phase, current, text):
            state["pending"] = {"key": interpreted["key"], "prompt": current["prompt"]}
            return f"Я понял твой ответ так: «{interpreted['text']}». Подтверди или исправь формулировку."
    if options_hidden(state):
        return with_prompt(state, "Не удалось проверить формулировку. Уточни результат или попроси показать варианты.")
    if phase == "solo" and status in ("disabled", "unavailable"):
        return with_prompt(state, "Разбор свободного текста сейчас недоступен. Сформулируй полный вывод и выбери эквивалентный вариант в списке вариантов.")
    if status == "unavailable":
        return with_prompt(state, "GigaChat сейчас недоступен. Выбери вариант в списке или повтори попытку.")
    return with_prompt(state, "Пока не удалось однозначно сопоставить ответ. Укажи элементы, узлы и вывод, или выбери вариант.")
