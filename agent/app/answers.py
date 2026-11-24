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


def branch_claims(raw, *, strict=False):
    text = norm(raw)
    if re.search(ELEMENT_NEGATION, text) or re.search(UNSURE, text):
        return {}
    components = r"[rl][1-9]\d*(?: (?:и )?[rl][1-9]\d*)*"
    directions = r"верх\w*|ниж\w*|сверху|снизу|наверху|вверху|внизу"
    pattern = (
        rf"(?P<names>{components}) (?:тоже )?(?:на )?(?P<direction>{directions})(?: вет\w*)?"
        rf"|(?P<leading_direction>{directions})(?: вет\w*)? (?P<following_names>{components})"
    )
    claims = {}
    matches = list(re.finditer(pattern, text))
    for match in matches:
        direction = match["direction"] or match["leading_direction"]
        branch = "upper" if direction.startswith(("верх", "сверху", "наверху", "вверху")) else "lower"
        names = set(COMPONENT.findall(match["names"] or match["following_names"]))
        claims.setdefault(branch, set()).update(names)
    if claims.get("upper", set()) & claims.get("lower", set()):
        return {}
    if set(COMPONENT.findall(text)) != set().union(*claims.values()):
        return {}
    if strict:
        remainder = re.sub(pattern, " ", text)
        remainder = re.sub(r"^(?:привет|здравствуйте|добрый день)\b", " ", remainder.strip())
        # Only the closed answer notation is eligible for a parser-only grade.
        # Additional prose must go through semantic verification as a whole.
        if re.sub(r"\bи\b|\s+", "", remainder):
            return {}
    return claims


def elements_match(raw, reference):
    text = norm(raw)
    return (
        set(COMPONENT.findall(text)) == set(COMPONENT.findall(norm(reference)))
        and not re.sub(r"\b[rl][1-9]\d*\b|\bи\b|\s+", "", text)
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


def resolve_ordinal_elements(text, components):
    roots = ("перв", "втор", "трет", "четверт", "пят", "шест",
             "седьм", "восьм", "девят", "десят", "одиннадцат", "двенадцат")
    result = norm(text)
    cardinals = ("один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать")
    for name, _, _ in components:
        number = int(re.search(r"\d+", name)[0])
        if 1 <= number <= len(cardinals):
            result = re.sub(r"\bномер (?:" + str(number) + "|" + cardinals[number - 1] + r")\b", name.lower(), result)
    for root, (name, _, _) in zip(roots, components):
        result = re.sub(r"\b" + root + r"\w*\b(?!\s+(?:вывод|контакт))", name.lower(), result)
    return result


def contradicts_all_elements(text, step, shape):
    if step not in ("series", "parallel"):
        return False
    if not re.search(r"\b(?:все|всех|кажд\w*|люб\w*)\b", norm(text)):
        return False
    # Claims about whole branches are distinct from claims about every element.
    if re.search(r"\b(?:ветв\w*|участ\w*)\b", norm(text)):
        return False
    names = {name for name, _, _ in shape.components}
    groups = shape.series if step == "series" else shape.parallel
    return not any(set(group) == names for group, _ in groups)


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
        notation = re.fullmatch(r"(?:[rl][1-9]\d* (?:[a-z]|неизвестно) (?:[a-z]|неизвестно) ?)+", answer_key(raw, "nodes"))
        return bool(notation and parsed) and parsed == node_claims(reference)

    # Keep every claim and qualification in a conclusion. Only known
    # synonyms are interchangeable here.
    return topology_key(raw) == topology_key(reference)
