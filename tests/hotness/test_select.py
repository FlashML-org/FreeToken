"""选点工具（freetoken.hotness.select）的回归：stats 校验、top-K 平局规则、
预算折算、pin list 落盘回读、覆盖率报告与 CLI 端到端。纯 CPU，无 torch 依赖
（--budget-gib 分支的延迟导入除外）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from freetoken.hotness.select import (
    coverage_report,
    load_stats,
    main,
    select_pins,
    slots_from_budget,
    write_pin_list,
)

L, E = 4, 16


def _write_stats(
    path: Path,
    counts: list[list[int]],
    model_path: str | None = None,
    schema_version: int = 1,
) -> Path:
    """在 tmp_path 下写一份合成 stats JSON，meta 可带 model_path。"""
    meta: dict = {
        "num_layers": len(counts),
        "num_experts": len(counts[0]),
        "top_k": 2,
        "total_tokens": sum(sum(row) for row in counts),
        "moe_strategy": "hybrid",
        "quant_format": "nvfp4",
        "model_path": model_path or "/models/fake",
        "duration_s": 1.0,
        "created_at": "2026-09-27T00:00:00+08:00",
    }
    payload = {"schema_version": schema_version, "meta": meta, "counts": counts}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _uniform_counts(layers: int = L, experts: int = E, base: int = 10) -> list[list[int]]:
    """每层计数 = base + expert_id，便于推断确定性的 top-K。"""
    return [[base + e for e in range(experts)] for _ in range(layers)]


# ---------------------------------------------------------------- select_pins


def test_select_pins_orders_desc_and_truncates():
    """计数互异时严格按计数降序排列，且只取前 K 个。"""
    counts = _uniform_counts()  # 每层计数 10..25，降序即 id 15,14,...,0
    pins = select_pins(counts, 4)
    assert pins == [[15, 14, 13, 12]] * L


def test_select_pins_tie_breaks_to_small_id():
    """平局必须取小 id：把 id 2 与 id 9 设成相同最大计数。"""
    counts = _uniform_counts()
    for row in counts:
        row[2] = row[9] = 100  # 并列最大
        row[5] = row[7] = 99  # 并列次大
    pins = select_pins(counts, 2)
    assert pins == [[2, 9]] * L
    pins4 = select_pins(counts, 4)
    assert pins4 == [[2, 9, 5, 7]] * L


def test_select_pins_all_tied_takes_lowest_ids():
    """全零计数（极端平局）应稳定取每层最小的 K 个 id。"""
    counts = [[0] * E for _ in range(L)]
    assert select_pins(counts, 5) == [list(range(5))] * L


def test_select_pins_rejects_k_covering_every_expert():
    """K >= 专家数拒绝：冷 bank 至少留 1 行，不能截成全量再交给建 bank。"""
    counts = _uniform_counts()
    with pytest.raises(ValueError, match="冷 bank"):
        select_pins(counts, E)
    with pytest.raises(ValueError, match="冷 bank"):
        select_pins(counts, 99)
    assert len(select_pins(counts, E - 1)[0]) == E - 1


def test_select_pins_rejects_bad_inputs():
    """K < 1 与非矩形 counts 必须抛 ValueError。"""
    counts = _uniform_counts()
    with pytest.raises(ValueError, match="slots_per_layer"):
        select_pins(counts, 0)
    ragged = _uniform_counts()
    ragged.append([1] * (E - 1))
    with pytest.raises(ValueError, match="一致"):
        select_pins(ragged, 2)


# ------------------------------------------------------------ slots_from_budget


def test_slots_from_budget_floors_and_enforces_minimum():
    """向下取整正确；预算折算不足 1 个专家时仍返回最小 1。"""
    gib = 1.0
    bytes_per_expert = 2**20
    assert slots_from_budget(gib, bytes_per_expert, 100) == 10  # 1024/100 = 10.24
    assert slots_from_budget(gib, bytes_per_expert, 1024) == 1  # 恰好整除
    assert slots_from_budget(gib, bytes_per_expert, 2000) == 1  # 0.51 -> 截断后仍最小 1
    assert slots_from_budget(0.5, 2**30, 1) == 1  # floor(0.5)=0 -> 1
    assert slots_from_budget(2.0, 2**28, 1) == 8  # 2*2**30/2**28 = 8，单层整除
    assert slots_from_budget(2.0, 2**28, 8) == 1  # 8/8 层 = 1，恰好整除


def test_slots_from_budget_never_exceeds_budget():
    """任何参数下 层数×K×每专家字节 不得超出预算（最小 1 除外）。"""
    total = 3.0 * 2**30
    k = slots_from_budget(3.0, 1234567, 48)
    assert k * 1234567 * 48 <= total


def test_slots_from_budget_rejects_bad_inputs():
    with pytest.raises(ValueError):
        slots_from_budget(0.0, 100, 4)
    with pytest.raises(ValueError):
        slots_from_budget(1.0, 0, 4)
    with pytest.raises(ValueError):
        slots_from_budget(1.0, 100, 0)


# ------------------------------------------------------------------ load_stats


def test_load_stats_accepts_valid_file(tmp_path: Path):
    """合法 stats 原样返回 meta 与 counts。"""
    counts = _uniform_counts()
    path = _write_stats(tmp_path / "stats.json", counts)
    data = load_stats(str(path))
    assert data["meta"]["num_layers"] == L
    assert data["meta"]["num_experts"] == E
    assert data["counts"] == counts


@pytest.mark.parametrize(
    "mutate, pattern",
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["meta"].update(num_layers=99), "不一致"),
        (lambda d: d["meta"].update(num_experts=0), "正整数"),
        (lambda d: d.update(counts={"0": [0] * 16}), "嵌套 list"),
        (lambda d: d["counts"][0].pop(), "不一致"),
        (lambda d: d["counts"][1].__setitem__(2, -1), "负数"),
        (lambda d: d["counts"][2].__setitem__(3, 1.5), "int"),
        (lambda d: d["counts"][3].__setitem__(4, True), "int"),
    ],
)
def test_load_stats_rejects_bad_files(tmp_path: Path, mutate, pattern):
    """schema 版本、形状、类型与数值非法时都必须带明确信息地抛 ValueError。"""
    counts = _uniform_counts()
    path = _write_stats(tmp_path / "stats.json", counts)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        load_stats(str(path))


def test_load_stats_missing_file(tmp_path: Path):
    with pytest.raises(ValueError, match="无法读取"):
        load_stats(str(tmp_path / "absent.json"))


# -------------------------------------------------------------- write_pin_list


def test_write_pin_list_roundtrip(tmp_path: Path):
    """写出的 pin list 读回后 schema 字段与内容逐项一致，且不留临时文件。"""
    counts = _uniform_counts()
    pins = select_pins(counts, 3)
    out = tmp_path / "pins.json"
    write_pin_list(str(out), L, E, 3, pins)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["num_layers"] == L
    assert data["num_experts"] == E
    assert data["per_layer_slots"] == 3
    assert data["pins"] == [
        {"layer": layer, "experts": pins[layer]} for layer in range(L)
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pins.json"]


def test_write_pin_list_rejects_inconsistent_pins(tmp_path: Path):
    """层数、每层长度、重复或越界专家 id 都必须拒绝。"""
    out = str(tmp_path / "pins.json")
    with pytest.raises(ValueError, match="num_layers"):
        write_pin_list(out, L, E, 2, [[0, 1]] * (L - 1))
    with pytest.raises(ValueError, match="slots_per_layer"):
        write_pin_list(out, L, E, 2, [[0, 1, 2]] * L)
    with pytest.raises(ValueError, match="重复"):
        write_pin_list(out, L, E, 2, [[0, 0]] * L)
    with pytest.raises(ValueError, match="越界"):
        write_pin_list(out, L, E, 2, [[0, E]] * L)


# ------------------------------------------------------------- coverage_report


def test_coverage_report_contains_key_numbers():
    """报告必须包含每层占比、全局命中率与零命中数的具体数值。"""
    counts = [[0] * E for _ in range(L)]
    for layer in range(L):
        counts[layer][0] = 50
        counts[layer][1] = 30
        counts[layer][2] = 20  # 未选中的最大计数项，压低 top-2 之外的流量
    pins = select_pins(counts, 2)  # 每层选中 id 0,1 -> 80/100 = 80.00%
    report = coverage_report(counts, pins)
    assert "80.00%" in report
    # 全局: 选中 (50+30)*4=320 / 总 (50+30+20)*4=400 = 80.00%
    assert "选中计数 320" in report
    assert "总计数 400" in report
    # 零命中: 每层 E-3=13 个 -> 52 / 64
    assert "52 / 64" in report


def test_coverage_report_zero_traffic_layer_shows_na():
    """整层零流量的占比应显示 n/a 而不是伪造的 0%。"""
    counts = [[0] * E, _uniform_counts(layers=1)[0]]
    pins = select_pins(counts, 2)
    report = coverage_report(counts, pins)
    assert "layer    0: n/a" in report


# ------------------------------------------------------------------ CLI 端到端


def test_cli_select_writes_pin_list(tmp_path: Path, capsys):
    """main() 端到端：写 stats -> select --slots -o -> 读回 pin list 断言。"""
    counts = _uniform_counts()
    for row in counts:
        row[10] = 500  # 每层无可争议的 top1
    stats = _write_stats(tmp_path / "stats.json", counts)
    out = tmp_path / "pins.json"

    code = main(["select", str(stats), "--slots", "3", "-o", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_layer_slots"] == 3
    assert data["pins"] == [{"layer": layer, "experts": [10, 15, 14]} for layer in range(L)]
    stdout = capsys.readouterr().out
    assert "全局加权命中率估计" in stdout  # 覆盖率报告打印到 stdout


def test_cli_select_without_output_prints_examples(tmp_path: Path, capsys):
    """不给 -o 时不写文件，只打印前几层选点示例。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts())
    code = main(["select", str(stats), "--slots", "2"])
    assert code == 0
    captured = capsys.readouterr()
    assert not list(tmp_path.glob("pins*.json"))
    assert "layer 0: [15, 14]" in captured.out


def test_cli_select_budget_gib_with_explicit_dims(tmp_path: Path):
    """--budget-gib 走 _BANK_BYTES_PER_EXPERT['bf16'] 折算（6*H*I 字节/专家）。"""
    hidden, intermediate = 64, 32
    bytes_per_expert = 6 * hidden * intermediate  # bf16: 3*I*H*2 = 12288
    # 预算 1e-4 GiB = 107374.18 字节 -> floor(107374.18/12288/4) = 2
    expected_k = math.floor(0.0001 * 2**30 / bytes_per_expert / L)
    assert expected_k == 2

    counts = _uniform_counts()
    stats = _write_stats(tmp_path / "stats.json", counts)
    out = tmp_path / "pins.json"
    code = main(
        [
            "select", str(stats), "--budget-gib", "0.0001",
            "--format", "bf16", "--hidden-size", str(hidden),
            "--moe-intermediate", str(intermediate), "-o", str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_layer_slots"] == expected_k
    assert all(len(pin["experts"]) == expected_k for pin in data["pins"])


def test_cli_select_budget_gib_reads_model_config(tmp_path: Path):
    """不传维度时从 stats meta.model_path 的 config.json 补齐。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"hidden_size": 64, "moe_intermediate_size": 32}), encoding="utf-8"
    )
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts(), model_path=str(model_dir))
    out = tmp_path / "pins.json"
    # 5e-5 GiB = 53687.09 字节 -> floor(53687.09/12288/4) = 1
    code = main(["select", str(stats), "--budget-gib", "0.00005", "--format", "bf16", "-o", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_layer_slots"] == 1


def test_cli_select_budget_gib_without_dims_fails_cleanly(tmp_path: Path, capsys):
    """模型目录没有 config.json 时报可操作的中文错误并返回 2。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts())
    code = main(["select", str(stats), "--budget-gib", "1.0", "--format", "bf16"])
    assert code == 2
    assert "hidden" in capsys.readouterr().err


def test_cli_select_unknown_format_fails_cleanly(tmp_path: Path, capsys):
    """--format 不在 _BANK_BYTES_PER_EXPERT 中时报出可选值。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts())
    code = main(
        ["select", str(stats), "--budget-gib", "1.0", "--format", "nope",
         "--hidden-size", "64", "--moe-intermediate", "32"]
    )
    assert code == 2
    assert "nvfp4" in capsys.readouterr().err


def test_cli_select_rejects_k_covering_every_expert(tmp_path: Path, capsys):
    """--slots 钉满一层时退出码 2，不写出 pin list。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts())
    out = tmp_path / "pins.json"
    code = main(["select", str(stats), "--slots", str(E), "-o", str(out)])
    assert code == 2
    assert "冷 bank" in capsys.readouterr().err
    assert not out.exists()


def test_cli_select_bad_stats_returns_two(tmp_path: Path, capsys):
    """stats 校验失败经 CLI 转成退出码 2 与 stderr 信息。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts(), schema_version=7)
    code = main(["select", str(stats), "--slots", "2"])
    assert code == 2
    assert "schema_version" in capsys.readouterr().err


def test_cli_slots_and_budget_are_mutually_exclusive(tmp_path: Path):
    """同时给 --slots 与 --budget-gib（或都不给）由 argparse 拒绝。"""
    stats = _write_stats(tmp_path / "stats.json", _uniform_counts())
    with pytest.raises(SystemExit) as both:
        main(["select", str(stats), "--slots", "2", "--budget-gib", "1.0"])
    assert both.value.code == 2
    with pytest.raises(SystemExit) as none:
        main(["select", str(stats)])
    assert none.value.code == 2
