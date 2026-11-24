"""Check learner-sourced assertions against the circuit, without model-issued grades."""
import re
from itertools import combinations

from .answers import COMPONENT, NODE, branch_claims, recall_matches, resolve_ordinal_elements
from .knowledge import CIRCUITS, norm


def names(text):
    return set(COMPONENT.findall(norm(text)))


def groups_key(groups):
    return frozenset(frozenset(group) for group in groups)


def evidence_chunks(text):
    pieces = [text]
    pieces += re.split(r"[.;!?\n]+", text)
    for piece in list(pieces):
        markers = list(re.finditer(r"\b(?:[rlрp][1-9]\d*|перв\w*|втор\w*|трет\w*|четверт\w*|пят\w*|шест\w*)\b", piece, re.I))
        pieces += [piece[match.start():markers[i + 1].start() if i + 1 < len(markers) else len(piece)]
                   for i, match in enumerate(markers)]
    cleaned = dict.fromkeys(piece.strip(" \t\n,;.!?«»\"") for piece in pieces if piece.strip())
    return {f"s{i}": piece for i, piece in enumerate(cleaned) if piece}


def question_chunks(text, chunks):
    result = []
    for quote in chunks.values():
        if re.search(r"[.;!\n]", quote):
            continue
        start = text.find(quote)
        ending = text[start + len(quote):].lstrip()
        if ending.startswith("?") or re.match(r"^(?:а |но |подожди,? )?(?:что|как|почему|зачем|кто|где|объясни|подскажи|помоги|правильно)\b", quote, re.I):
            result.append(quote)
    return result


def resolve_evidence(value, chunks):
    if isinstance(value, list):
        return [resolve_evidence(item, chunks) for item in value]
    if isinstance(value, dict):
        return {key: ([chunks.get(part, "") for part in item] if key == "retracted_evidence" and isinstance(item, list)
                      else chunks.get(item, "") if key.endswith("evidence") and isinstance(item, str)
                      else resolve_evidence(item, chunks)) for key, item in value.items()}
    return value


def literal_overall(evidence):
    """Only an explicit, unambiguous type in the learner's own conclusion counts."""
    first = re.split(r"[.;:,\n]|\bпотому\b|\bпоскольку\b|\bтак как\b", evidence, maxsplit=1, flags=re.I)[0]
    text = norm(first)
    if re.search(r"недостаточ\w* (?:данн\w*|информац\w*)|(?:не|нельзя|невозможно) определ\w*", text):
        return "unknown"
    if re.search(r"\b(?:не|нет|если|или)\b", text):
        return None
    kinds = {kind for kind, pattern in (("mixed", r"смешан\w*|комбинирован\w*"),
             ("series", r"последоват\w*"), ("parallel", r"параллел\w*")) if re.search(pattern, text)}
    return next(iter(kinds)) if len(kinds) == 1 else None


def sourced_solution(solution, text, shape):
    """Remove extraction artefacts lacking learner provenance; they cannot earn credit."""
    result = dict(solution)
    if "elements" in result:
        item = result["elements"]
        quote = literal_evidence(item, text, shape)
        if not quote or not item.get("names") or not {norm(name) for name in item["names"]} <= names(quote):
            result["elements"] = {}
    if result.get("overall_evidence") and result["overall_evidence"] in text:
        declared = literal_overall(result["overall_evidence"])
        if declared:
            result["overall"] = declared
    for key in ("terminals", "relations", "branches", "universals"):
        kept = []
        for item in solution.get(key, []):
            quote = literal_evidence(item, text, shape)
            if not quote:
                continue
            if key == "terminals":
                # A closed literal pair is stronger evidence than model-filled
                # slots. Keep all records so conflicting pairs still fail below.
                pair = re.fullmatch(r"([rlрp][1-9]\d*)\s*:\s*([a-z])\s*[–—-]\s*([a-z])",
                                    item.get("evidence", ""), re.I)
                if pair and norm(pair[1]) == norm(item["element"]):
                    item = {**item, "nodes": [pair[2].upper(), pair[3].upper()]}
                valid = names(quote) == {norm(item["element"])} and (len(NODE.findall(quote)) == 2 or "?" in item.get("nodes", []) and len(NODE.findall(quote)) == 1 and re.search(r"неизвест\w*|скрыт\w*|не видно", quote))
            elif key == "universals":
                predicate = {"same_terminals": r"узл\w*|вывод\w*|подключ\w*", "series": r"последоват\w*", "parallel": r"параллел\w*"}[item["relation"]]
                valid = bool(re.search(r"\b(?:все|всех|кажд\w*|люб\w*)\b", quote) and re.search(predicate, quote))
            else:
                elements = item["elements"] if key == "branches" else [name for group in item["groups"] for name in (group.split("+") if isinstance(group, str) else group)]
                valid = bool(elements) and {norm(element) for element in elements} <= names(quote)
            if valid:
                kept.append(item)
        result[key] = kept
    return result


def uncovered_physics(text, solution, question_text=""):
    # A citation is a source range, not a licence to accept every assertion in it.
    covered = set()
    entity = r"[rlрp][1-9]\d*|[a-z]"
    operator = r"узл\w*|вывод\w*|контакт\w*|подключ\w*|соедин\w*|общ\w*"
    def mark(quote, allowed=None, limit=None):
        if not quote:
            return
        start = 0
        while (start := text.find(quote, start)) >= 0:
            selected = quote if limit is None else quote[:limit]
            if allowed is None:
                covered.update(range(start, start + len(selected)))
            else:
                for match in re.finditer(r"\b(?:" + allowed + r")\b", selected, re.I):
                    covered.update(range(start + match.start(), start + match.end()))
            start += len(quote)
    for item in solution.get("terminals", []):
        quote = item.get("evidence", "")
        letters = list(re.finditer(r"\b[a-z]\b", quote, re.I))
        components = list(re.finditer(r"\b[rlрp][1-9]\d*\b", quote, re.I))
        endpoint = max([match.end() for match in letters + components] or [0])
        unknown = list(re.finditer(r"скрыт\w*|неизвест\w*|не видно", quote, re.I))
        if "?" in item.get("nodes", []) and unknown:
            endpoint = max(endpoint, unknown[-1].end())
        mark(quote, entity + "|" + operator + r"|скрыт\w*|неизвест\w*", endpoint)
    for item in solution.get("branches", []):
        mark(item.get("evidence", ""), entity + r"|ветв\w*|ветк\w*|верх\w*|ниж\w*|сверху|снизу|наверху|внизу|вверху")
    for item in solution.get("relations", []):
        predicate = r"последоват\w*" if item["relation"] == "series" else r"параллел\w*"
        mark(item.get("evidence", ""), entity + "|" + operator + "|" + predicate + r"|ответв\w*|ветв\w*|провод\w*")
        if item["relation"] == "series" and absence_condition(item.get("evidence", "")):
            mark(item.get("evidence", ""), r"не|нет|без|отсутств\w*")
    for item in solution.get("universals", []):
        predicate = {"same_terminals": operator, "series": r"последоват\w*", "parallel": r"параллел\w*"}[item["relation"]]
        mark(item.get("evidence", ""), entity + "|" + predicate + r"|все|всех|кажд\w*|люб\w*|одинак\w*|разн\w*|ветв\w*|участ\w*")
        if item.get("value") is False:
            mark(item.get("evidence", ""), r"не|нет|неодинак\w*")
    mark(solution.get("elements", {}).get("evidence", ""), entity + r"|элемент\w*|резистор\w*|ламп\w*|проводник\w*")
    mark(solution.get("action_evidence", ""))
    mark(solution.get("no_relation_evidence", ""), r"нет|отсутств\w*|последоват\w*|параллел\w*|участ\w*")
    overall = solution.get("overall_evidence", "")
    mark(overall, r"смешан\w*|последоват\w*|параллел\w*|определ\w*|скрыт\w*|неизвест\w*|вывод\w*|контакт\w*|" + entity)
    if solution.get("overall") == "unknown":
        mark(overall, r"не|нельзя|невозможно")
        for clause in re.split(r"[.;!?\n]+", text):
            if re.search(r"может|возможно|если", clause, re.I):
                mark(clause, entity + r"|вывод\w*|контакт\w*|подключ\w*|узл\w*|если|или")
    for quote in solution.get("retracted_evidence", []):
        action = list(re.finditer(r"зач[её]рк\w*|отзыв\w*|отмен\w*", quote, re.I))
        if not action:
            continue
        end = action[-1].end()
        if action[0].start() == 0:
            separator = re.search(r",|\b(?:но|теперь|зато)\b", quote[end:], re.I)
            end = end + separator.start() if separator else len(quote)
        mark(quote, limit=end)
    mark(question_text)
    physical = r"[rlрp][1-9]\d*|[a-z]|узл\w*|вывод\w*|подключ\w*|последоват\w*|параллел\w*|ответв\w*|ветв\w*|одинак\w*|изгиб\w*|созда\w*|явля\w*|ток\w*|сопротивл\w*|напряжен\w*|мощност\w*"
    markers = r"\b(?:" + physical + r"|все|всех|кажд\w*|если|или|кроме|неверн\w*|неправд\w*|ложн\w*|не|нет|без|ни)\b"
    for match in re.finditer(markers, text, re.I):
        word = match.group().lower()
        if word in ("не", "нет", "без", "ни"):
            if word == "не" and re.match(r"\s+(?:уверен\w*|знаю|понимаю|помню)\b", text[match.end():], re.I):
                continue
            left = max([text.rfind(separator, 0, match.start()) for separator in ".;!?\n"])
            right = min([pos for separator in ".;!?\n" if (pos := text.find(separator, match.end())) >= 0] or [len(text)])
            if not re.search(r"\b(?:" + physical + r")\b", text[left + 1:right], re.I):
                continue
        if not set(range(match.start(), match.end())) <= covered:
            return True
    return False


def expected_groups(shape, relation):
    return {groups_key([names(group) for group in groups]): frozenset(node.lower() for node in nodes)
            for groups, nodes in getattr(shape, relation)}


def section_ends(shape, section):
    section = set(section)
    selected = [(name.lower(), a.lower(), b.lower() if b else None)
                for name, a, b in shape.components if name.lower() in section]
    if len(selected) != len(section) or any(b is None for _, _, b in selected):
        return None
    inside = {node for _, a, b in selected for node in (a, b)}
    outside = {node.lower() for name, a, b in shape.components if name.lower() not in section
               for node in (a, b) if node}
    boundary = inside & (outside | {"a", "b"})
    return frozenset(boundary) if len(boundary) == 2 else frozenset()


def relation_properties(shape, relation, groups):
    if len(groups) < 2 or sum(map(len, groups)) != len(set().union(*groups)):
        return False, frozenset()
    ends = [section_ends(shape, group) for group in groups]
    if any(pair is None for pair in ends):
        return None, frozenset()
    if any(len(pair) != 2 for pair in ends):
        return False, frozenset()
    if relation == "parallel":
        return len(set(ends)) == 1, ends[0]
    adjacency = {}
    for pair in ends:
        a, b = pair
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(neighbours) > 2 for neighbours in adjacency.values()):
        return False, frozenset()
    if sum(len(neighbours) == 1 for neighbours in adjacency.values()) != 2:
        return False, frozenset()
    seen, stack = set(), [next(iter(adjacency))]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(adjacency[node])
    if len(seen) != len(adjacency):
        return False, frozenset()
    joins = frozenset(node for node, neighbours in adjacency.items() if len(neighbours) == 2)
    members = set().union(*groups)
    outside = {node.lower() for name, a, b in shape.components if name.lower() not in members
               for node in (a, b) if node}
    if joins & (outside | {"a", "b"}):
        return False, joins
    return True, joins


def absence_condition(evidence):
    """An actual absence predicate, not an unrelated 'не' and 'провод'."""
    for clause in re.split(r"[.;!?\n]", evidence):
        text = norm(clause)
        direct = re.search(r"\b(?:нет|без|отсутств\w*)\s+(?:\w+\s+){0,3}(?:ответв\w*|разветв\w*|ветв\w*)", text)
        reverse = re.search(r"\b(?:ответв\w*|разветв\w*|ветв\w*)\s+(?:\w+\s+){0,2}(?:нет|отсутств\w*)\b", text)
        outgoing = re.search(r"\b(?:провод\w*|ветв\w*|ответв\w*)\b.{0,50}\bне\s+(?:\w+\s+){0,2}(?:отход\w*|уход\w*|идет|выход\w*)\b", text)
        if direct or reverse or outgoing:
            scope = "between_groups" if re.search(r"\bмежду\b.*(?:участ\w*|блок\w*|ними|этими)", text) else "at_node"
            return scope
    return None


def literal_evidence(item, text, shape):
    quote = item.get("evidence", "")
    if not quote or quote not in text:
        return None
    return norm(resolve_ordinal_elements(quote, shape.components))


def assess_solution(case_id, task, solution, text, *, question_text=""):
    """Return complete/incomplete/contradiction plus a bounded learner-facing reason.

    Every accepted assignment has literal input provenance. Relations are compared
    with the computed topology; absent entities or justification cannot be supplied
    by the knowledge base or by a semantic 'equivalent' verdict.
    """
    shape = CIRCUITS[case_id]
    known = {name.lower() for name, _, _ in shape.components}
    step = "action" if task.get("recalling") else task.get("step", "branches")
    expected_pairs = {name.lower(): frozenset((a.lower(), b.lower() if b else "?"))
                      for name, a, b in shape.components}
    if step == "action":
        evidence = solution.get("action_evidence", "")
        if (solution.get("action") == task["step"] and evidence in text and evidence
                and recall_matches(task["step"], evidence)):
            return "complete", ""
        return "incomplete", "Сначала назови действие, которое выполняешь."
    unknown = any(b is None for _, _, b in shape.components)
    assertions = text.replace(question_text, "", 1) if question_text else text
    if re.search(r"\b(?:все|всех|кажд\w*|люб\w*)\b", norm(assertions)) and not solution.get("universals"):
        return "incomplete", "Уточни общее утверждение обо всех элементах: оно тоже должно соответствовать схеме."
    for item in solution.get("universals", []):
        quote = literal_evidence(item, text, shape)
        if quote is None:
            return "incomplete", "Перепиши утверждение о всех элементах явно."
        negative = bool(re.search(r"\bне\b|неодинак\w*|\bразн\w*", quote))
        if item["value"] == negative:
            return "incomplete", "Уточни, утверждаешь ли ты этот признак для всех элементов или отрицаешь его."
        relation = item["relation"]
        domain = item.get("domain", "branches" if re.search(r"ветв\w*|участ\w*", quote) else "elements")
        if domain != "elements":
            candidate = next((entry for entry in solution.get("relations", []) if entry["relation"] == relation), None)
            if not candidate:
                return "incomplete", "Укажи, какие целые участки входят в общее утверждение."
            groups = [{norm(name) for name in (group.split("+") if isinstance(group, str) else group)} for group in candidate["groups"]]
            actual, _ = relation_properties(shape, relation, groups) if relation != "same_terminals" else (None, None)
            if actual is None:
                return "incomplete", "Для общего утверждения не хватает известных связей."
        else:
            actual = (len(set(expected_pairs.values())) == 1 if relation == "same_terminals" else
                  any(set().union(*group) == known and all(len(part) == 1 for part in group)
                      for group in expected_groups(shape, relation)))
        if unknown or actual != item["value"]:
            return "contradiction", "Общее утверждение обо всех элементах не соответствует их связям на схеме. Проверь его отдельно."
    if step == "branches":
        actual = {}
        for item in solution.get("branches", []):
            quote = literal_evidence(item, text, shape)
            elements = {norm(element) for element in item["elements"]}
            if quote is None or not elements <= names(quote) or not elements <= known:
                return "incomplete", "Укажи обозначения элементов на каждой ветви."
            actual.setdefault(item["position"], set()).update(elements)
        branch_groups, _ = shape.parallel[0]
        expected = {key: names(group) for key, group in zip(("upper", "lower"), branch_groups)}
        explicit = branch_claims(text)
        if explicit and explicit != actual:
            return "contradiction", "Извлечённое расположение расходится с твоей записью. Уточни верхнюю и нижнюю ветви."
        if actual.get("upper", set()) & actual.get("lower", set()):
            return "contradiction", "Один элемент указан сразу в двух разных ветвях. Уточни расположение."
        if any(not values <= expected.get(key, set()) for key, values in actual.items()):
            return "contradiction", "В расположении элементов есть ошибка. Проверь каждую ветвь по рисунку."
        return ("complete", "") if actual == expected else ("incomplete", "Ответ пока неполный. Укажи расположение всех элементов, включая оставшуюся ветвь.")
    pairs = {}
    for item in solution.get("terminals", []):
        quote = literal_evidence(item, text, shape)
        element = norm(item["element"])
        nodes = frozenset(norm(node) if node != "?" else "?" for node in item["nodes"])
        if quote is None or names(quote) != {element} or element not in expected_pairs:
            return "incomplete", "Запиши подключения каждого элемента отдельно."
        if re.search(r"\b(?:не|нет|без|кроме|или|если)\b|неверн\w*|неправд\w*|ложн\w*", quote):
            return "incomplete", "Запиши выбранные подключения утвердительно, без альтернатив или условий."
        quoted_nodes = set(NODE.findall(quote))
        if "?" in nodes and re.search(r"неизвест\w*|скрыт\w*|не видно", quote):
            quoted_nodes.add("?")
        if len(item["nodes"]) != 2 or nodes != quoted_nodes:
            return "incomplete", "Укажи два вывода элемента или явно отметь скрытый вывод."
        if element in pairs and pairs[element] != nodes:
            return "contradiction", "Для одного элемента указаны разные подключения. Уточни, какой ответ оставляешь."
        if nodes != expected_pairs[element]:
            if "?" in expected_pairs[element]:
                return "incomplete", f"Второй вывод {element.upper()} скрыт. По рисунку нельзя установить конкретный узел его подключения."
            return "contradiction", "Одна из указанных пар узлов не совпадает со схемой. Проследи оба вывода этого элемента."
        pairs[element] = nodes
    if step == "elements":
        item = solution.get("elements", {})
        quote = literal_evidence(item, text, shape)
        extracted = {norm(name) for name in item.get("names", [])}
        if quote is not None and re.search(r"\b(?:нет|не|без|кроме|или|если)\b|неверн\w*|неправд\w*|ложн\w*", quote):
            return "incomplete", "Перечисли присутствующие элементы утвердительно, без отрицания или альтернатив."
        if quote is not None and extracted == names(quote) == known:
            return "complete", ""
        return "incomplete", "Перечисли все обозначенные элементы рассматриваемого участка."
    if step == "nodes":
        return ("complete", "") if pairs == expected_pairs else ("incomplete", "Пока указаны не все подключения. Допиши пары узлов остальных элементов.")
    found = {"series": set(), "parallel": set()}
    series_joins, series_members, parallel_members = set(), set(), {}
    for item in solution.get("relations", []):
        quote = literal_evidence(item, text, shape)
        relation = item["relation"]
        groups = [{norm(name) for name in (group.split("+") if isinstance(group, str) else group)}
                  for group in item["groups"]]
        if len(groups_key(groups)) != len(groups):
            return "incomplete", "Уточни, какие отдельные элементы или целые участки ты сравниваешь."
        key = groups_key(groups)
        nodes = frozenset(norm(node) for node in item["nodes"])
        if quote is None or not groups or not set().union(*groups) <= names(quote):
            return "incomplete", "Назови элементы выбранных участков своими обозначениями."
        if re.search(r"\bне\s+(?:\w+\s+){0,2}(?:последоват\w*|параллел\w*)", quote):
            return "incomplete", "Уточни, какое соединение ты утверждаешь: отрицание нельзя считать положительным ответом."
        truth, required_nodes = relation_properties(shape, relation, groups)
        if truth is None:
            return "incomplete", "Часть подключений неизвестна, поэтому такой вывод пока нельзя подтвердить."
        if not truth:
            return "contradiction", "Пока не получилось подтвердить состав и соединение участков. Запиши каждый целый участок отдельно и укажи его внешние узлы."
        if nodes != required_nodes or not nodes <= set(NODE.findall(quote)):
            return "incomplete", "Ты назвал участки. Теперь укажи их общие узлы, чтобы обосновать соединение."
        if relation == "series":
            # Evidence of the required condition must come from the learner, not
            # from an inferred meaning of the word 'последовательное'.
            if unknown:
                return "incomplete", "Вывод зависит от скрытой связи. Рассмотри оба возможных подключения."
            scope = absence_condition(item["evidence"])
            if not item.get("no_extra_branch") or not scope:
                return "incomplete", "Осталось объяснить, есть ли отдельная ветвь между выбранными участками."
            if scope == "at_node" and any(sum(node in (a.lower(), b.lower() if b else None) for _, a, b in shape.components) > 2 for node in nodes):
                return "contradiction", "В указанном узле есть разветвление внутри составного участка. Уточни, что между целыми участками нет отдельной ветви."
            series_joins.update(nodes)
            series_members.update(set().union(*groups))
        else:
            parallel_members.setdefault(nodes, set()).update(set().union(*groups))
        found[relation].add(key)
    def covered(relation):
        if relation == "series":
            expected_nodes = {node.lower() for _, nodes in shape.series for node in nodes}
            expected_members = set().union(*(names(group) for groups, _ in shape.series for group in groups)) if shape.series else set()
            return bool(shape.series) and expected_nodes <= series_joins and expected_members <= series_members
        return bool(shape.parallel) and all(
            set().union(*(names(group) for group in groups)) <= parallel_members.get(frozenset(node.lower() for node in nodes), set())
            for groups, nodes in shape.parallel
        )
    if step in found:
        expected = set(expected_groups(shape, step))
        absence = solution.get("no_relation_evidence", "")
        explicit_absence = (not unknown and solution.get("no_relation") == step and absence and absence in text
                            and re.search(r"\bнет\b|\bотсутств\w*", norm(absence))
                            and re.search(r"последоват\w*" if step == "series" else r"параллел\w*", norm(absence)))
        if covered(step) or not expected and explicit_absence:
            return "complete", ""
        return "incomplete", "Проверь, все ли участки этого вида ты назвал, и обоснуй каждый."
    if step == "overall":
        expected_type = "unknown" if unknown else "mixed" if shape.series and shape.parallel else "series" if shape.series else "parallel"
        if literal_overall(solution.get("overall_evidence", "")) != solution.get("overall"):
            return "incomplete", "Запиши общий тип соединения явно: он должен совпадать с твоим пояснением."
        if solution.get("overall") != expected_type:
            return "contradiction", "Общий вывод пока не соответствует связям на схеме."
        if not solution.get("overall_evidence") or solution["overall_evidence"] not in text:
            return "incomplete", "Сформулируй общий вывод своими словами."
        if unknown:
            evidence = solution["overall_evidence"]
            if any(name.lower() in norm(evidence) for name, _, b in shape.components if b is None) and re.search(r"скрыт\w*|неизвест\w*|не видно", norm(evidence)):
                return "complete", ""
            return "incomplete", "Объясни, как вывод зависит от скрытого подключения."
        if all(covered(relation) if getattr(shape, relation) else not found[relation] for relation in found):
            return "complete", ""
        return "incomplete", "Обоснуй общий вывод найденными последовательными и параллельными участками."
    return "incomplete", "Запиши решение текущего шага."


def solution_schema(step, components, node_labels, chunks):
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    def array(items):
        return {"type": "array", "items": items}
    def enum(values):
        return {"type": "string", "enum": list(values)}
    evidence = {"type": "string", "enum": ["", *chunks], "description": "ID фрагмента evidence_chunks, содержащего это утверждение. Пустая строка если утверждения нет."}
    elements = array(enum(components))
    nodes = array(enum([*node_labels, "?"]))
    revoked = [key for key, quote in chunks.items() if not re.search(r"[.;!?\n]", quote)
               and re.search(r"зачерк\w*|отзыв\w*|отмен\w*", norm(quote))]
    properties = {"retracted_evidence": array(enum(revoked)) if revoked else {"type": "array", "maxItems": 0, "items": {"type": "string"}},
                  "universals": array(obj({
        "relation": enum(("same_terminals", "series", "parallel")), "domain": enum(("elements", "branches", "selected_groups")),
        "value": {"type": "boolean"}, "evidence": evidence,
    }))}
    if step == "branches":
        placements = {}
        for element in components:
            available = [key for key, quote in chunks.items()
                         if element.lower() in names(resolve_ordinal_elements(quote, [(name, None, None) for name in components]))]
            placements[element] = obj({"position": enum(("upper", "lower", "not_stated")),
                                       "evidence": {"type": "string", "enum": ["", *available]}})
        properties["placements"] = obj(placements)
    elif step == "action":
        properties.update(action=enum(("none", "elements", "nodes", "series", "parallel", "overall")), action_evidence=evidence)
    elif step == "elements":
        properties["elements"] = obj({"names": elements, "evidence": evidence})
    elif step == "nodes":
        variants = []
        for element in components:
            available = []
            for key, raw_quote in chunks.items():
                quote = resolve_ordinal_elements(raw_quote, [(name, None, None) for name in components])
                if names(quote) == {element.lower()} and (len(NODE.findall(norm(quote))) == 2
                        or len(NODE.findall(norm(quote))) == 1 and re.search(r"неизвест\w*|скрыт\w*|\?", quote, re.I)):
                    available.append(key)
            variants.append(obj({"element": enum([element]), "nodes": nodes,
                                 "evidence": {"type": "string", "enum": ["", *available]}}))
        properties["terminals"] = array({"anyOf": variants})
    else:
        sections = ["+".join(group) for size in range(1, len(components) + 1)
                    for group in combinations(components, size)]
        properties["relations"] = array(obj({
            "relation": enum(("series", "parallel")), "groups": {**array(enum(sections)), "minItems": 2, "uniqueItems": True,
                "description": "Одиночные элементы — отдельные значения. Целый участок из нескольких элементов обозначается через +. Не повторяй одно значение дважды."}, "nodes": nodes,
            "no_extra_branch": {"type": "boolean", "description": "true только если ученик явно обосновал отсутствие отдельной ветви между участками."},
            "evidence": evidence,
        }))
        properties["no_relation"] = enum(("none", "series", "parallel"))
        properties["no_relation_evidence"] = evidence
        if step == "overall":
            properties.update(overall=enum(("none", "series", "parallel", "mixed", "unknown")), overall_evidence=evidence)
    return obj(properties)
