"""Small explicit generation gates; not a standardized accuracy benchmark."""

import json
import re


def first_answer(text):
    for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.split(marker, 1)[0]
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def json_ok(text):
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return False
    try:
        return json.loads(match.group()) == {"name": "test", "count": 3}
    except json.JSONDecodeError:
        return False


CASES = [
    (
        "moe_cn",
        "请用两句话解释什么是混合专家（MoE）模型。",
        lambda t: "专家" in t and any(w in t for w in ("路由", "激活", "选择", "门控", "分配", "决定")),
    ),
    ("addition", "What is 137 + 286? Reply with the number only.", lambda t: "423" in t),
    ("division", "What is 144 divided by 12? Reply with the number only.", lambda t: "12" in t),
    (
        "python",
        "Write a Python function named add that returns the sum of two arguments. Output code only.",
        lambda t: "def add" in t and "return" in t and "+" in t,
    ),
    ("sky", "In one sentence, explain why the sky appears blue.", lambda t: "blue" in t.lower() and "scatter" in t.lower()),
    ("json", 'Return only a JSON object with keys name and count, using name "test" and count 3.', json_ok),
    ("capital_cn", "中国的首都是哪里？只回答城市名称。", lambda t: "北京" in t),
    ("decimal", "Which is larger, 0.9 or 0.11? Answer with just the larger number.", lambda t: "0.9" in t),
]
