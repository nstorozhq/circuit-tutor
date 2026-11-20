import json
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

DATA = json.loads((Path(__file__).resolve().parents[1] / "knowledge_base/lesson.json").read_text())
CASES = {case["id"]: case for case in DATA["cases"]}
STAGES = ("guided", "spoken", "solo", "test")
COURSE = [case for stage in STAGES for case in DATA["cases"] if case["stage"] == stage]
STEPS = ("elements", "nodes", "series", "parallel", "overall")
PROMPTS = {
    "elements": "Какие элементы входят в рассматриваемый участок?",
    "nodes": "Между какими узлами подключён каждый элемент?",
    "series": "Какие элементы или участки соединены последовательно? Назови общий узел и объясни, есть ли там ответвление.",
    "parallel": "Какие элементы или участки соединены параллельно? Укажи оба общих узла.",
    "overall": "Как соединён участок в целом? Обоснуй вывод по схеме."
}
STEP_NAMES = {
    "elements": "Элементы", "nodes": "Узлы", "series": "Последовательные участки",
    "parallel": "Параллельные участки", "overall": "Вывод"
}
HINTS = {
    "elements": ("Сосчитай подписанные элементы в каждой ветви.", "Перечисли обозначения рядом с прямоугольниками."),
    "nodes": ("Проследи провод от каждого вывода до буквы узла.", "Веди взгляд вдоль провода. Изгиб не создаёт нового узла."),
    "series": ("Проверь промежуточный узел между выбранными элементами.", "Если от общего узла отходит ещё одна ветвь, эти элементы не образуют простую последовательную цепочку."),
    "parallel": ("Сравни оба узла подключения выбранных участков.", "Проследи каждый участок целиком. Одного общего узла для параллельного соединения недостаточно."),
    "overall": ("Вспомни, какие последовательные и параллельные участки ты нашёл.", "Если вывод зависит от скрытой связи, укажи, какой вывод элемента не виден.")
}

@dataclass
class Trace:
    components: list
    series: list
    parallel: list
    wires: list
    boxes: list
    junctions: list


def trace_circuit(root):
    shape = Trace([], [], [], [], [], [])
    inner_nodes = iter("CDEFGHIJKLMNOPQRSTUVWXYZ")

    def line(x1, y1, x2, y2):
        if (x1, y1) != (x2, y2):
            shape.wires.append([round(v, 2) for v in (x1, y1, x2, y2)])

    def height(tree):
        if isinstance(tree, str) or "hidden" in tree:
            return 1
        if "s" in tree:
            return max(height(child) for child in tree["s"])
        return sum(height(child) for child in tree["p"])

    def names(tree):
        if isinstance(tree, str):
            return tree
        if "hidden" in tree:
            return tree["hidden"]
        op = "s" if "s" in tree else "p"
        separator = "–" if op == "s" else " ∥ "
        group = separator.join(names(item) for item in tree[op])
        return f"({group})"

    def visit(tree, a, b, x1, x2, y1, y2):
        mid = (y1 + y2) / 2
        if isinstance(tree, str) or "hidden" in tree:
            hidden = isinstance(tree, dict)
            name = tree["hidden"] if hidden else tree
            x = (x1 + x2) / 2
            shape.components.append([name, a, None if hidden else b])
            shape.boxes.append({"name": name, "x": round(x, 2), "y": round(mid, 2), "hidden": hidden})
            line(x1, mid, x - 21, mid)
            line(x + 21, mid, x + 29 if hidden else x2, mid)
            return
        if "s" in tree:
            items = tree["s"]
            labels = [a] + [next(inner_nodes) for _ in items[1:]] + [b]
            xs = [x1 + (x2 - x1) * i / len(items) for i in range(len(items) + 1)]
            shape.series.append(([names(item) for item in items], labels[1:-1]))
            for i, label in enumerate(labels[1:-1], 1):
                shape.junctions.append({"name": label, "x": round(xs[i], 2), "y": round(mid, 2)})
            for i, item in enumerate(items):
                visit(item, labels[i], labels[i + 1], xs[i], xs[i + 1], y1, y2)
            return
        items = tree["p"]
        left, right = x1 + 12, x2 - 12
        weights = [height(item) for item in items]
        bounds = [y1 + (y2 - y1) * sum(weights[:i]) / sum(weights) for i in range(len(items) + 1)]
        centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(items))]
        line(x1, mid, left, mid)
        line(right, mid, x2, mid)
        line(left, min(centers[0], mid), left, max(centers[-1], mid))
        line(right, min(centers[0], mid), right, max(centers[-1], mid))
        if not any(isinstance(item, dict) and "hidden" in item for item in items):
            shape.parallel.append(([names(item) for item in items], [a, b]))
        for i, item in enumerate(items):
            visit(item, a, b, left, right, bounds[i], bounds[i + 1])

    shape.junctions += [{"name": "A", "x": 22, "y": 99}, {"name": "B", "x": 338, "y": 99}]
    visit(root, "A", "B", 22, 338, 28, 170)
    return shape


def alternatives(correct, distractors, hint, seed, custom=None):
    choices = [{"key": "correct", "text": correct, "feedback": ""}]
    for index, wrong in enumerate(dict.fromkeys(distractors)):
        if wrong != correct:
            choices.append({"key": f"d{index}", "text": wrong, "feedback": (custom or {}).get(wrong, hint)})
    Random(seed).shuffle(choices)
    return choices


def question(case, step):
    shape = CIRCUITS[case["id"]]
    components = shape.components
    names = [item[0] for item in components]
    hidden = next((name for name, _, b in components if b is None), None)
    nodes = "; ".join(f"{name}: {a}–{b or '?'}" for name, a, b in components)
    series = "; ".join(
        f"{' и '.join(group)} соединены последовательно. "
        + (f"Нет ответвлений в узлах {', '.join(points)}."
           if not any(item.startswith("(") for item in group)
           else f"В узлах {', '.join(points)} между этими участками нет отдельной ветви.")
        for group, points in shape.series
    )
    if hidden and shape.series:
        group, points = shape.series[0]
        common = ", ".join(points)
        series = (
            f"Недостаточно данных. {' и '.join(group)} соединены последовательно, "
            f"если скрытый вывод {hidden} не подключён к {common}. "
            f"Если этот вывод подключён к {common}, там есть ответвление."
        )
    parallel = "; ".join(
        f"{' и '.join(group)} подключены к одним узлам {' и '.join(points)}."
        for group, points in shape.parallel
    )
    reference = {
        "elements": ", ".join(names),
        "nodes": nodes,
        "series": series or "Последовательных участков нет.",
        "parallel": parallel or (
            f"Недостаточно данных: не видно, к какому узлу подключён второй вывод {hidden}."
            if hidden else "Параллельных участков нет."
        ),
        "overall": (
            f"Не определить однозначно: второй вывод {hidden} скрыт."
            if hidden else (
                f"Смешанное. {series} {parallel}"
                if shape.series and shape.parallel else
                "Последовательное: все элементы в одной цепочке без ответвлений."
                if shape.series else "Параллельное: все ветви подключены к A и B."
            )
        )
    }[step]
    first, second, last = names[0], names[1], names[-1]
    one_node = f"{first} и {last} параллельны, потому что у них есть общий узел."
    wrong_nodes = "; ".join(
        f"{name}: {a}–{'C' if b == 'B' else 'B'}" if i == 0 else f"{name}: {a}–{b or '?'}"
        for i, (name, a, b) in enumerate(components)
    )
    distractors = {
        "elements": [", ".join(names[:-1]), ", ".join([*names, "источник тока"]), ", ".join([*names[1:], "соединительный провод"])],
        "nodes": [wrong_nodes, "; ".join(f"{name}: A–B" for name in names),
                  "; ".join(f"{name}: ?" for name in names)],
        "series": [f"{first} и {second} последовательны, потому что стоят рядом.",
                   "Все элементы последовательны, если они внутри одной схемы.",
                   "Последовательность определяется только числом элементов."],
        "parallel": [one_node, "Параллельны все элементы, которые нарисованы друг под другом.",
                     "На схеме точно нет параллельных участков, если у одного элемента скрыт вывод."
                     if hidden else "Любые две ветви с общим началом параллельны независимо от их концов."],
        "overall": ["Смешанное, потому что в схеме несколько элементов.",
                    "Последовательное, потому что элементы соединены проводами.",
                    "Параллельное, потому что на рисунке есть загнутый провод."]
    }[step]
    if step == "parallel" and case["id"] == "C01":
        specific = "R1 и R3 имеют общий узел A, но их вторые выводы подключены к C и B. Сравни пары узлов."
    else:
        specific = HINTS[step][1]
    choices = alternatives(reference, distractors, HINTS[step][0], case["id"] + step, {one_node: specific})
    hints = HINTS[step]
    if hidden and step == "series":
        hints = (
            "Может ли скрытый вывод быть подключён к общему узлу элементов?",
            "Сравни случай с ответвлением в общем узле и случай без него.",
        )
    return {"step": step, "prompt": PROMPTS[step], "choices": choices, "reference": reference, "hints": hints}


CIRCUITS = {case["id"]: trace_circuit(case["circuit"]) for case in DATA["cases"]}
GOAL_DISTRACTORS = {
    "Измерять температуру проводника.": "Температуру измеряют термометром. Здесь мы рассматриваем связи на схеме.",
    "Вычислять мощность любого прибора.": "Для вычисления мощности нужны числовые данные. Здесь мы рассматриваем связи на схеме.",
    "Рисовать схему без анализа связей.": "В этом задании схема уже дана. Нужно разобраться, как её элементы связаны между собой.",
}
INTRO = {
    "goal": {
        "prompt": "Какое действие мы будем тренировать с помощью этих знаний?",
        "choices": alternatives(
            "Определять способ соединения элементов в электрической схеме.",
            GOAL_DISTRACTORS,
            "Вспомни, что позволяют узнать признаки последовательного и параллельного соединений.",
            "goal", GOAL_DISTRACTORS)
    },
    "method": {
        "prompt": "Предложи способ решения. С чего начать и как проверить вывод?",
        "choices": alternatives(
            "Выделить элементы, определить их узлы, найти последовательные и параллельные участки, затем сделать вывод.",
            ["Сначала посчитать элементы и определить вид по их числу.",
             "Сравнить расположение элементов на рисунке.",
             "Найти хотя бы один общий узел и назвать соединение параллельным."],
            "Начни с элементов и их выводов. Затем проверь признаки обоих видов соединения.", "method")
    },
    "compare": {
        "prompt": "Сравни свою первую попытку с ответом учителя. Что получилось?",
        "choices": [
            {"key": "same", "text": "Мой вывод совпал с ответом учителя."},
            {"key": "different", "text": "Мой вывод или обоснование отличались."},
            {"key": "unclear", "text": "Пока трудно объяснить, где мы разошлись."},
        ],
    }
}

ACTION_NAMES = {
    "elements": "Выделяю элементы участка.",
    "nodes": "Определяю узлы подключения.",
    "series": "Ищу последовательные участки.",
    "parallel": "Ищу параллельные участки.",
    "overall": "Делаю вывод о виде соединения.",
}


def recall_question(step):
    return {
        "step": step,
        "prompt": "Назови следующее действие по памяти.",
        "choices": alternatives(
            ACTION_NAMES[step], [name for key, name in ACTION_NAMES.items() if key != step][:3],
            "Вспомни порядок действий или попроси показать варианты.", "recall" + step,
        ),
        "hints": ("Вспомни порядок: элементы, узлы, последовательные участки, параллельные участки, вывод.",),
    }


def norm(value):
    return re.sub(r"[^\w]+", " ", value.lower().replace("ё", "е")).strip()


def lookup(text):
    query = f" {norm(text)} "
    ranked = sorted(
        ((sum(len(tag) for tag in item["tags"] if f" {tag} " in query), item)
         for item in DATA["concepts"]),
        key=lambda pair: -pair[0],
    )
    return [item for rank, item in ranked if rank][:2]


def reference_answer(case, full=False):
    order = STEPS if full else STEPS[2:]
    return "\n\n".join(
        f"{STEP_NAMES[step]} — {question(case, step)['reference']}" for step in order
    )
