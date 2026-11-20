"""Conservative checks for typed answers before a model suggestion can affect progress."""

import re

from .knowledge import norm

RECALL_VERBS = {
    "elements": r"(?:выдел\w*|перечисл\w*|наз\w*|определ\w*|найд\w*) (?:все )?(?:элемент\w*|проводник\w*|резистор\w*|ламп\w*)",
    "nodes": r"(?:определ\w*|найд\w*|укаж\w*|отмеч\w*|выдел\w*|наз\w*) (?:все )?(?:узл\w*|точки подключ\w*)",
    "series": r"(?:найд\w*|ищ\w*|определ\w*|выдел\w*|провер\w*) последовательн\w* (?:участ\w*|соедин\w*|элемент\w*)",
    "parallel": r"(?:найд\w*|ищ\w*|определ\w*|выдел\w*|провер\w*) параллельн\w* (?:участ\w*|ветв\w*|соедин\w*|элемент\w*)",
    "overall": r"(?:дел\w*|сдел\w*|формулир\w*|сформулир\w*) (?:общий )?вывод|(?:определ\w*|наз\w*) (?:вид|тип) соединени\w*",
}
COMPONENT = re.compile(r"\b[rl][1-9]\d*\b")
NODE = re.compile(r"\b[a-z]\b")
ELEMENT_NEGATION = r"\b(?:не|кроме|без|исключ\w*|нет|или)\b"
UNSURE = r"\b(?:не знаю|не уверен\w*|возможно|может быть|или)\b"


def answer_key(text, step):
    if step == "nodes":
        text = text.replace("?", " неизвестно ")
    return norm(text)


def recall_matches(step, text):
    prefix = r"(?:(?:я|сначала|затем|теперь|далее|первым делом|следующее действие) )*"
    return bool(re.match(prefix + "(?:" + RECALL_VERBS[step] + r")\b", norm(text)))


def branch_claims(raw):
    text = norm(raw)
    if re.search(ELEMENT_NEGATION, text) or re.search(UNSURE, text):
        return {}
    components = r"[rl][1-9]\d*(?: (?:и )?[rl][1-9]\d*)*"
    directions = r"верх\w*|ниж\w*|сверху|снизу"
    pattern = (
        rf"(?P<names>{components}) (?:на )?(?P<direction>{directions})(?: вет\w*)?"
        rf"|(?P<leading_direction>{directions})(?: вет\w*)? (?P<following_names>{components})"
    )
    claims = {}
    for match in re.finditer(pattern, text):
        direction = match["direction"] or match["leading_direction"]
        branch = "upper" if direction.startswith(("верх", "сверху")) else "lower"
        names = set(COMPONENT.findall(match["names"] or match["following_names"]))
        if branch in claims and claims[branch] != names:
            return {}
        claims[branch] = names
    if set(COMPONENT.findall(text)) != set().union(*claims.values()):
        return {}
    return claims


def elements_match(raw, reference):
    text = norm(raw)
    return (
        set(COMPONENT.findall(text)) == set(COMPONENT.findall(norm(reference)))
        and not re.search(ELEMENT_NEGATION, text)
        and not re.search(r"\b(?:источник\w*|батаре\w*|провод|соединительн\w* провод\w*)\b", text)
    )


def node_claims(text):
    parts = re.split(r"\b([rl][1-9]\d*)\b", answer_key(text, "nodes"))
    result = {}
    for name, claim in zip(parts[1::2], parts[2::2]):
        found = NODE.findall(claim)
        if "неизвестно" in claim:
            found.append("?")
        if name in result or len(found) != 2:
            return {}
        result[name] = frozenset(found)
    return result


def topology_key(text):
    value = norm(text)
    for pattern, equivalent in (
        (r"\bразветвлени\w*\b", "ответвлений"),
        (r"\bотсутствуют ответвлени\w*\b", "нет ответвлений"),
        (r"\bсоединяются последовательно\b", "соединены последовательно"),
        (r"\bвключены последовательно\b", "соединены последовательно"),
        (r"\bк одним и тем же узлам\b", "к одним узлам"),
        (r"\bне хватает информации\b", "недостаточно данных"),
        (r"\bсоединение смешанное\b", "смешанное"),
    ):
        value = re.sub(pattern, equivalent, value)
    return value


def allows_correct_suggestion(phase, task, raw):
    """Apply a deliberately narrow rule set. Unclear text can still use a visible option."""
    text = norm(raw)
    if re.search(UNSURE, text):
        return False
    if phase == "method":
        return all(re.search(pattern, text) for pattern in (
            r"элемент", r"узл|точк[аи] подключ", r"последовательн", r"параллельн", r"вывод",
        )) and not re.search(r"\bне\b", text)

    step = task["step"]
    reference = task["reference"]
    if step == "elements":
        return elements_match(raw, reference)
    if step == "nodes":
        parsed = node_claims(raw)
        return bool(parsed) and parsed == node_claims(reference) and not re.search(r"\bне\b", text)

    # Keep every claim and qualification in a conclusion. Only known
    # synonyms are interchangeable here.
    return topology_key(raw) == topology_key(reference)
