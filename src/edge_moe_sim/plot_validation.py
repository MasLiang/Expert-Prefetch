"""Small dependency-free validation helpers for characterization plotting."""
from __future__ import annotations

from typing import Mapping, Sequence
import csv
from pathlib import Path


def require_plot_data(rows: Sequence[Mapping[str, object]], required: Sequence[str]) -> None:
    """Reject empty/malformed plot data instead of writing a misleading blank plot."""
    if not rows:
        raise ValueError("plot data is empty")
    missing = [name for name in required if any(name not in row or row[name] in (None, "") for row in rows)]
    if missing:
        raise ValueError(f"plot data has missing required columns/values: {sorted(set(missing))}")


def validate_line_group(rows: Sequence[Mapping[str, object]], *, sweep: str,
                        fixed: Sequence[str]) -> None:
    """Ensure a line connects exactly one intended dimension under fixed context."""
    require_plot_data(rows, [sweep, *fixed])
    varying = [name for name in fixed if len({row[name] for row in rows}) != 1]
    if varying:
        raise ValueError(f"line group varies uncontrolled dimensions: {varying}")
    if len({row[sweep] for row in rows}) < 2:
        raise ValueError(f"line group needs at least two {sweep} values")


def assert_single_sweep_dimension(rows: Sequence[Mapping[str, object]], *, sweep: str,
                                  context: Sequence[str]) -> None:
    """Alias with a publication-oriented name for figure construction code."""
    validate_line_group(rows, sweep=sweep, fixed=context)


def assert_unique_x_per_line(rows: Sequence[Mapping[str, object]], *, x_column: str,
                             line_group_columns: Sequence[str], figure_name: str,
                             debug_dir: str | Path | None = None) -> None:
    """Reject duplicate x observations and emit differing-column diagnostics."""
    require_plot_data(rows, [x_column, *line_group_columns])
    buckets: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[column] for column in (*line_group_columns, x_column))
        buckets.setdefault(key, []).append(row)
    problems = []
    for key, group in buckets.items():
        if len(group) > 1:
            differing = {column: sorted({str(row.get(column, "")) for row in group})
                         for column in set().union(*(row.keys() for row in group))
                         if len({str(row.get(column, "")) for row in group}) > 1}
            problems.append({"figure": figure_name, "line_group": key[:-1],
                             "x_value": key[-1], "row_count": len(group),
                             "differing_columns": repr(differing)})
    if problems:
        if debug_dir is not None:
            path = Path(debug_dir); path.mkdir(parents=True, exist_ok=True)
            output = path / f"duplicate_x_{figure_name}.csv"
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(problems[0])); writer.writeheader(); writer.writerows(problems)
            output.with_suffix(".md").write_text("# Duplicate x diagnostics\n\n" + "\n".join(str(row) for row in problems), encoding="utf-8")
        raise ValueError(f"duplicate x observations for {figure_name}: {len(problems)} groups")
