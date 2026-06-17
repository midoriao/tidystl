from __future__ import annotations

from pathlib import Path

from tests._helpers.jsonl_io import read_jsonl, write_jsonl


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "gt.jsonl"
    records = [
        {"name": "b_case", "time": [0.0, 1.0], "robustness": [1.0, -2.0]},
        {"name": "a_case", "time": [0.0], "robustness": [3.0]},
    ]
    write_jsonl(path, records)
    loaded = read_jsonl(path)
    # sorted by name on write
    assert [r["name"] for r in loaded] == ["a_case", "b_case"]
    assert loaded[1]["robustness"] == [1.0, -2.0]


def test_write_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "gt.jsonl"
    write_jsonl(path, [{"name": "x", "time": [0.0], "robustness": [0.0]}])
    text = path.read_text()
    assert text.count("\n") == 1
    assert text.startswith('{"name": "x"')


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gt.jsonl"
    path.write_text('{"name": "x", "time": [0.0], "robustness": [0.0]}\n\n')
    assert len(read_jsonl(path)) == 1
