#!/usr/bin/env python3
"""
Evaluation runner with iteration support and CSV metrics tracking.
"""

import argparse
import asyncio
import csv
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from fast_agent import FastAgent, ConversationSummary
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.mcp.prompt_serialization import save_messages
from test_eval_assertions import (
    ASSERTIONS_TOTAL,
    validate_with_metrics,
    ValidationResult,
)

# Create the application
fast = FastAgent("fast-agent example", ignore_unknown_args=True)

# CSV fieldnames for results tracking
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
    "llm_time_ms",
    "tool_time_ms",
    "turns",
]

# Stable CSV path for accumulating results across batches
CSV_PATH = Path("runs") / "regraded_results.csv"


def categorize_tool_calls(
    tool_call_map: dict[str, int],
    tool_error_map: dict[str, int],
) -> dict[str, int]:
    """Categorize tool calls and errors into MCP vs execute.

    MCP tools have the pattern 'server__tool' (double underscore).
    Execute/skill tools typically don't have '__' in their name.

    Returns dict with keys: mcp_calls, mcp_errors, execute_calls, execute_errors
    """
    mcp_calls = 0
    mcp_errors = 0
    execute_calls = 0
    execute_errors = 0

    for tool_name, count in tool_call_map.items():
        if "__" in tool_name:
            mcp_calls += count
        else:
            execute_calls += count

    for tool_name, count in tool_error_map.items():
        if "__" in tool_name:
            mcp_errors += count
        else:
            execute_errors += count

    return {
        "mcp_calls": mcp_calls,
        "mcp_errors": mcp_errors,
        "execute_calls": execute_calls,
        "execute_errors": execute_errors,
    }

# Files to exclude when collecting artifacts
EXCLUDE_FILES = {
    "agent.py",
    "test_eval_assertions.py",
    "fastagent.config.yaml",
    "fastagent.secrets.yaml",
}

default_instruction = """You are a helpful AI Agent.

{{serverInstructions}}

{{agentSkills}}

{{file_silent:AGENTS.md}}

{{env}}

The current date is {{currentDate}}."""


def reset_skills_repo() -> None:
    """Reset the skills repository to a clean state."""
    skills_path = Path("../skills").resolve()
    print(f"Resetting skills repo at {skills_path}...")
    subprocess.run(["git", "checkout", "."], cwd=skills_path, check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=skills_path, check=True, capture_output=True)
    print("Skills repo reset complete.")


def collect_artifacts(run_folder: Path) -> list[str]:
    """Move generated .py and .yaml files to the run folder.

    Returns list of moved file names.
    """
    moved_files = []
    for pattern in ["*.yaml", "*.py"]:
        for f in Path(".").glob(pattern):
            if f.name in EXCLUDE_FILES:
                continue
            dest = run_folder / f.name
            shutil.move(str(f), str(dest))
            moved_files.append(f.name)
    return moved_files


def parse_args() -> argparse.Namespace:
    """Parse command line arguments, ignoring unknown args (e.g., --model for FastAgent)."""
    parser = argparse.ArgumentParser(description="Run evaluation iterations with metrics tracking")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="olmo_7b_evaluations.yaml",
        help="Name of the output YAML file to validate (default: olmo_7b_evaluations.yaml)",
    )
    args, _ = parser.parse_known_args()
    return args


# Define the agent
@fast.agent(
    name="eval_skill",
    skills=["../skills/"],
    servers=["huggingface"],
    instruction=default_instruction,
)
async def main():
    args = parse_args()
    num_runs = args.runs
    output_filename = args.output_file

    # Create timestamped batch folder for artifacts
    batch_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
    batch_folder = Path("runs") / batch_id
    batch_folder.mkdir(parents=True, exist_ok=True)

    # Check if CSV exists (to determine if we need to write header)
    csv_exists = CSV_PATH.exists()

    print(f"\n{'='*60}")
    print(f"Starting evaluation batch: {num_runs} run(s)")
    print(f"Batch ID: {batch_id}")
    print(f"Artifacts folder: {batch_folder}")
    print(f"Results CSV: {CSV_PATH}")
    print(f"{'='*60}\n")

    results: list[dict] = []

    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not csv_exists:
            writer.writeheader()

        for i in range(1, num_runs + 1):
            print(f"\n--- Run {i}/{num_runs} ---")
            run_timestamp = datetime.now().isoformat()

            # Reset skills repo before each run
            try:
                reset_skills_repo()
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to reset skills repo: {e}")

            # Create run subfolder
            run_folder = batch_folder / f"run_{i}"
            run_folder.mkdir(exist_ok=True)

            # Initialize row with defaults
            row = {
                "batch_id": batch_id,
                "run_number": i,
                "model": "",
                "timestamp": run_timestamp,
                "passed": False,
                "assertions_passed": 0,
                "assertions_total": ASSERTIONS_TOTAL,
                "metrics_count": 0,
                "benchmarks_found": "",
                "tokens": 0,
                "conversation_span_ms": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "mcp_calls": 0,
                "mcp_errors": 0,
                "execute_calls": 0,
                "execute_errors": 0,
                "error_message": "",
                "llm_time_ms": 0.0,
                "tool_time_ms": 0.0,
                "turns": 0,
            }

            try:
                async with fast.run() as agent:
                    eval_agent = agent.eval_skill
#                    await agent.interactive()
                    # Run the evaluation task
                    await eval_agent.generate(load_prompt(Path("build_olmo_yaml.md")))

                    # Get model name
                    model_name = eval_agent.llm.model_name
                    row["model"] = model_name or ""

                    # Get conversation metrics
                    summary = ConversationSummary(messages=eval_agent.message_history)
                    row["conversation_span_ms"] = summary.conversation_span_ms
                    row["llm_time_ms"] = round(summary.total_elapsed_time_ms, 2)
                    # Tool time is whatever is left in the wall-clock span after LLM time
                    row["tool_time_ms"] = round(
                        max(summary.conversation_span_ms - row["llm_time_ms"], 0.0), 2
                    )
                    row["turns"] = summary.user_message_count

                    # Get tool call metrics
                    row["tool_calls"] = summary.tool_calls
                    row["tool_errors"] = summary.tool_errors

                    # Categorize by MCP vs execute
                    tool_categories = categorize_tool_calls(
                        summary.tool_call_map,
                        summary.tool_error_map,
                    )
                    row["mcp_calls"] = tool_categories["mcp_calls"]
                    row["mcp_errors"] = tool_categories["mcp_errors"]
                    row["execute_calls"] = tool_categories["execute_calls"]
                    row["execute_errors"] = tool_categories["execute_errors"]

                    # Get token usage
                    if eval_agent.llm.usage_accumulator:
                        row["tokens"] = eval_agent.llm.usage_accumulator.cumulative_billing_tokens

                    # Save conversation history
                    history_file = run_folder / "conversation.json"
                    save_messages(eval_agent.message_history, str(history_file))

                # Validate the output file
                output_path = Path(output_filename)
                validation: ValidationResult = validate_with_metrics(output_path)

                row["passed"] = validation.passed
                row["assertions_passed"] = validation.assertions_passed
                row["assertions_total"] = validation.assertions_total
                row["metrics_count"] = validation.metrics_count
                row["benchmarks_found"] = ",".join(validation.benchmarks_found)
                row["error_message"] = validation.error_message or ""

                status = "PASSED" if validation.passed else "FAILED"
                print(f"Run {i}: {status} ({validation.assertions_passed}/{validation.assertions_total} assertions)")

            except Exception as e:
                row["error_message"] = str(e)
                print(f"Run {i}: ERROR - {e}")

            # Collect artifacts (move generated files to run folder)
            moved = collect_artifacts(run_folder)
            if moved:
                print(f"Collected artifacts: {moved}")

            # Write row to CSV and flush
            writer.writerow(row)
            csvfile.flush()
            results.append(row)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed_count}/{num_runs}")
    print(f"Batch ID: {batch_id}")
    print(f"Artifacts: {batch_folder}")
    print(f"Results CSV: {CSV_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
