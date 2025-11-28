#!/usr/bin/env python3
"""
Regrade existing run artifacts with stricter validation and richer timing stats.

This reads the existing runs/results.csv, re-validates each run's YAML with the
updated assertions, and recomputes LLM vs tool time plus turn counts from the
conversation logs. A new CSV is written to runs/regraded_results.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from fast_agent import ConversationSummary
from fast_agent.mcp.prompt_serialization import load_messages

from test_eval_assertions import (
    ASSERTIONS_TOTAL,
    EXPECTED_METRICS,
    ValidationResult,
    validate_with_metrics,
)

RESULTS_PATH = Path("runs") / "results.csv"
OUTPUT_PATH = Path("runs") / "regraded_results.csv"
DEFAULT_OUTPUT_FILE = "olmo_7b_evaluations.yaml"

# Existing fieldnames from agent.py plus the new timing columns.
FIELDNAMES = [
    "batch_id",
    "run_number",
    "model",
    "timestamp",
    "passed",
    "assertions_passed",
    "assertions_total",
    "metrics_count",
    "benchmarks_found",
    "tokens",
    "conversation_span_ms",
    "tool_calls",
    "tool_errors",
    "mcp_calls",
    "mcp_errors",
    "execute_calls",
    "execute_errors",
    "error_message",
    # New fields for regrading
    "llm_time_ms",
    "tool_time_ms",
    "turns",
]


def find_yaml(run_folder: Path) -> Optional[Path]:
    """Locate the YAML output for a run."""
    candidate = run_folder / DEFAULT_OUTPUT_FILE
    if candidate.exists():
        return candidate
    yamls = [p for p in run_folder.glob("*.yaml")]
    return yamls[0] if yamls else None


def summarize_conversation(conversation_path: Path) -> tuple[float, float, int]:
    """Return (llm_time_ms, tool_time_ms, turns) from a conversation log."""
    if not conversation_path.exists():
        return 0.0, 0.0, 0

    messages = load_messages(str(conversation_path))
    summary = ConversationSummary(messages=messages)

    llm_time_ms = round(summary.total_elapsed_time_ms, 2)
    span_ms = summary.conversation_span_ms
    tool_time_ms = round(max(span_ms - llm_time_ms, 0.0), 2)
    turns = summary.user_message_count
    return llm_time_ms, tool_time_ms, turns


def regrade():
    assert RESULTS_PATH.exists(), f"Source results CSV not found at {RESULTS_PATH}"

    with open(RESULTS_PATH, newline="") as infile, open(OUTPUT_PATH, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in reader:
            batch_id = row["batch_id"]
            run_number = row["run_number"]
            run_folder = Path("runs") / batch_id / f"run_{run_number}"
            conversation_path = run_folder / "conversation.json"
            yaml_path = find_yaml(run_folder)

            llm_time_ms, tool_time_ms, turns = summarize_conversation(conversation_path)

            # Start with the original row, but override with new grades and timing.
            new_row = {**row}
            new_row["llm_time_ms"] = llm_time_ms
            new_row["tool_time_ms"] = tool_time_ms
            new_row["turns"] = turns

            if yaml_path is None:
                new_row.update(
                    {
                        "passed": False,
                        "assertions_passed": 0,
                        "assertions_total": ASSERTIONS_TOTAL,
                        "metrics_count": 0,
                        "benchmarks_found": "",
                        "error_message": "Output YAML not found",
                    }
                )
                writer.writerow(new_row)
                continue

            validation: ValidationResult = validate_with_metrics(
                yaml_path,
                expected_metrics=EXPECTED_METRICS,
            )

            new_row.update(
                {
                    "passed": validation.passed,
                    "assertions_passed": validation.assertions_passed,
                    "assertions_total": validation.assertions_total,
                    "metrics_count": validation.metrics_count,
                    "benchmarks_found": ",".join(validation.benchmarks_found),
                    "error_message": validation.error_message or "",
                }
            )

            writer.writerow(new_row)

    print(f"Regraded results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    regrade()
