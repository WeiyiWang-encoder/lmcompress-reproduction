"""Runners: one resumable, crash-safe path from data to reported ratios.

Design constraints that come from the environment this was built in (a machine
that loses power):

* progress is committed to disk **per window**, so at most one forward pass is
  ever repeated;
* results are appended one JSON object per unit, so a run that dies at 90 % still
  leaves usable measurements for the 90 %;
* predictor weights are loaded once and released explicitly, because a 4-bit
  Llama and a bGPT must never both sit on a 12 GB card.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence

import numpy as np

from .data import Unit
from .engine import UnitResult, Window, append_jsonl, ChunkCursor, measure_stream


def plan_windows(
    n_items: int,
    capacity: int,
    predictor_name: str = "",
    has_bos: bool = False,
    budget: int | None = None,
) -> List[Window]:
    """Split ``n_items`` symbols into scoring windows.

    Bookkeeping for sequential models, verified at run time by ``measure_unit``:

    * a forward pass over ``L`` fed positions yields ``L - 1`` predictions, since
      position ``p`` predicts ``p + 1``;
    * to score ``n`` tokens a window must therefore FEED ``n + 1`` of them -- the
      extra one is lookahead, needed so the last scored token has a successor to
      be predicted from.  It is fed but never scored.

    So window ``k`` scores exactly ``[start_k, stop_k)`` and consecutive windows
    tile the payload with neither a gap nor an overlap.  ``budget`` is the
    per-window token count, normally supplied by
    ``LlamaTokenPredictor.window_budget()``, which calibrates it against the model
    rather than trusting this arithmetic (see the note there).

    bGPT is different: its patch grid is a hard limit and its windows are
    independent, exactly as in the reference implementation.
    """
    if n_items <= 0 or capacity <= 0:
        return []

    if predictor_name == "bgpt":
        cap = capacity
        windows: List[Window] = []
        idx = 0
        while idx < n_items:
            stop = min(n_items, idx + cap)
            windows.append(Window(start=idx, stop=stop, prior=idx, offset=0))
            idx = stop
        from .models import bgpt_offsets

        for window, offset in zip(windows, bgpt_offsets(n_items, capacity)):
            window.offset = offset
        return windows

    # ``budget`` comes from the model when available.  The fallback keeps the
    # planner usable in isolation (tests, dry runs) and matches Llama-3-8B at any
    # context length: BOS and the causal shift consume the same single position,
    # so first and later windows score the same number of tokens.
    per_window = int(budget) if budget else max(1, capacity - 2)

    # ``window.stop`` is the exclusive end of the SCORED range; ``iter_steps`` is
    # given one further token as lookahead.  Hence the blocks below tile the
    # payload exactly.
    blocks: List[int] = []
    remaining = n_items
    while remaining > 0:
        take = per_window
        if take >= remaining:
            blocks.append(remaining)
            break
        blocks.append(take)
        remaining -= take

    windows = []
    start = 0
    for block in blocks:
        windows.append(Window(start=start, stop=start + block, prior=start))
        start += block
    return windows


def _unit_steps(predictor, unit: Unit, window: Window) -> Iterator:
    """Dispatch to the right predictor method for this window.

    ``window.stop`` is the exclusive end of the *scored* range.  For the
    sequential model we also pass the single following token: ``iter_steps`` needs
    it as lookahead so that the last scored token has a successor position to be
    predicted from, and it skips the first prediction so nothing is counted
    twice.  ``stop_at`` tells it exactly where to stop scoring.
    """
    if predictor.name == "llama":
        payload = unit.payload[window.start : min(window.stop + 1, len(unit.payload))]
        return predictor.iter_steps(payload, prior=window.prior, stop_at=window.stop)
    payload = unit.payload[window.start : window.stop]
    if predictor.name == "bgpt":
        return predictor.iter_steps(payload, offset=window.offset)
    return predictor.iter_steps(payload)


def measure_unit(
    predictor,
    unit: Unit,
    cursor_dir: str,
    verify_roundtrip: bool = False,
    commit_every: int = 1,
) -> UnitResult:
    """Measure one unit, checkpointing after every window."""
    is_bytes = predictor.name == "bgpt"
    has_bos = bool(getattr(predictor, "bos_id", None))
    budget_fn = getattr(predictor, "window_budget", None)
    per_window = budget_fn(has_bos) if callable(budget_fn) else None
    windows = plan_windows(
        len(unit.payload),
        predictor.window_capacity,
        predictor.name,
        has_bos=has_bos,
        budget=per_window,
    )
    cursor_path = os.path.join(cursor_dir, f"{_safe(unit.name)}.progress.json")
    cursor = ChunkCursor(
        cursor_path,
        total_chunks=len(windows),
        meta={"unit": unit.name, "predictor": predictor.name, "capacity": predictor.window_capacity},
    )

    result = UnitResult(raw_bytes=0)
    result.coded_bits = float(cursor.state["coded_bits"])
    result.coded_symbols = int(cursor.state["coded_symbols"])

    def scored_length(index: int, window: Window) -> int:
        """Payload tokens this window scores.

        ``window.stop`` is the *exclusive end of the scored range*; the block size
        is derived from the planner, so the totals are exact by construction
        rather than inferred.  The final window also has one uncovered token,
        which the predictor supplies as a sentinel.
        """
        return max(0, min(window.stop, len(unit.payload)) - window.start)

    expected_symbols = sum(scored_length(i, w) for i, w in enumerate(windows))

    t0 = time.perf_counter()
    for index, window in enumerate(windows):
        if index < cursor.done_chunks:
            continue
        wres, _ = measure_stream(
            _unit_steps(predictor, unit, window), verify_roundtrip=verify_roundtrip
        )
        cursor.commit(
            index,
            bits=wres.coded_bits,
            symbols=wres.coded_symbols,
            raw_bytes=0,
        )
        result.coded_bits += wres.coded_bits
        result.coded_symbols += wres.coded_symbols
        result.ac_bits += wres.ac_bits
        if verify_roundtrip:
            result.extra.setdefault("roundtrip", []).append(wres.extra.get("roundtrip"))

    # The denominator is the real data, never the (possibly padded) symbol count.
    # Coding fewer symbols than the payload means part of the input was never
    # measured, which would silently flatter the ratio, so it is reported.
    if result.coded_symbols != len(unit.payload):
        result.extra["uncoded_symbols"] = len(unit.payload) - result.coded_symbols
    result.raw_bytes = int(unit.original_bytes)
    result.wall_seconds = time.perf_counter() - t0
    result.extra.update(
        {
            "unit": unit.name,
            "predictor": predictor.name,
            "symbols": len(unit.payload),
            "windows": len(windows),
        }
    )
    cursor.mark_finished(result.as_dict())
    return result


def _safe(name: str) -> str:
    keep = "-_.() "
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)[:150]


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #


def run_batch(
    predictor,
    units: Sequence[Unit],
    results_path: str,
    cursor_dir: str,
    verify_roundtrip: bool = False,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> List[dict]:
    """Measure every unit and append one JSON record per unit.

    Already-finished units are skipped, which is what makes re-running the same
    command after a power cut cheap and safe.
    """
    os.makedirs(cursor_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)

    todo = list(units)
    if limit is not None:
        todo = todo[:limit]

    done: set[str] = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["unit"])
                except (json.JSONDecodeError, KeyError):
                    continue

    records: List[dict] = []
    for i, unit in enumerate(todo, 1):
        if unit.name in done:
            log(f"[{i}/{len(todo)}] skip (already measured): {unit.name}")
            continue
        log(f"[{i}/{len(todo)}] coding {unit.name} ({len(unit.payload)} symbols)")
        t0 = time.perf_counter()
        result = measure_unit(
            predictor, unit, cursor_dir, verify_roundtrip=verify_roundtrip
        )
        record = result.as_dict()
        record["unit"] = unit.name
        record.update({k: v for k, v in unit.meta.items() if k != "residual"})
        records.append(record)
        append_jsonl(results_path, record)
        log(
            f"    rate={record['compression_rate_pct']:.4f}%  "
            f"ratio={record['compression_ratio']:.4f}  "
            f"bpb={record['bits_per_byte']:.4f}  "
            f"wall={time.perf_counter() - t0:.1f}s"
        )
    return records


def summarize(records: Sequence[dict], denominator_key: str = "raw_bytes") -> dict:
    """Aggregate per-unit records into the paper's per-dataset numbers."""
    usable = [r for r in records if r.get(denominator_key)]
    if not usable:
        return {}
    orig = sum(int(r[denominator_key]) for r in usable)
    bits = sum(float(r["coded_bits"]) for r in usable)
    return {
        "units": len(usable),
        "original_bytes": orig,
        "coded_bits": bits,
        "bits_per_byte": bits / orig,
        "compression_rate_pct": 100.0 * (bits / 8.0) / orig,
        "compression_ratio": orig / (bits / 8.0),
        "wall_seconds": sum(float(r.get("wall_seconds", 0.0)) for r in usable),
    }
