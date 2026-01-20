#!/usr/bin/env python3
"""
Evaluation runner with iteration support and CSV metrics tracking.
"""

import argparse
import asyncio
import contextlib
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from fast_agent import ConversationSummary, FastAgent
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import get_session_manager
from test_eval_assertions import (
    ASSERTIONS_TOTAL,
    ValidationResult,
    validate_with_metrics,
)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "fastagent.config.yaml"
PROMPT_SOURCE = ROOT_DIR / "build_olmo_yaml.md"
AGENTS_SOURCE = ROOT_DIR / "AGENTS.md"
SKILLS_REPO_URL = "https://github.com/huggingface/skills.git"
SKILLS_REPO_COMMIT = "fe044dc129e33aca7c2edc0084f02a7119b4109f"
SKILL_NAMES = (
    "hugging-face-evaluation",
    "hugging-face-evaluation-manager",
)
SKILL_MANIFEST_CANDIDATES = [
    Path("skills") / "hugging-face-evaluation" / "SKILL.md",
    Path("hf_model_evaluation")
    / "skills"
    / "hugging-face-evaluation-manager"
    / "SKILL.md",
    Path("hf_model_evaluation") / "skills" / "hugging-face-evaluation" / "SKILL.md",
]

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
    "session_id",
    "session_history_file",
]

# Stable CSV path for accumulating results across batches
CSV_PATH = ROOT_DIR / "runs" / "regraded_results.csv"


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


# Files to exclude when copying artifacts into run workspaces
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


@contextlib.contextmanager
def run_in_workspace(path: Path):
    """Temporarily change the working directory to the given path."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def copy_prompt_assets(workspace: Path) -> None:
    """Copy prompt inputs needed by the agent into the workspace."""
    if not PROMPT_SOURCE.exists():
        raise FileNotFoundError(f"Prompt source missing: {PROMPT_SOURCE}")
    if not AGENTS_SOURCE.exists():
        raise FileNotFoundError(f"AGENTS.md missing: {AGENTS_SOURCE}")
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPT_SOURCE, workspace / PROMPT_SOURCE.name)
    shutil.copy2(AGENTS_SOURCE, workspace / AGENTS_SOURCE.name)


def _clone_skills_repo(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    subprocess.run(
        ["git", "clone", "--no-checkout", SKILLS_REPO_URL, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", SKILLS_REPO_COMMIT],
        check=True,
        capture_output=True,
        text=True,
    )


def _find_skill_manifest(skills_repo_dir: Path) -> Path:
    for candidate in SKILL_MANIFEST_CANDIDATES:
        manifest = skills_repo_dir / candidate
        if manifest.exists():
            return manifest

    for manifest in skills_repo_dir.rglob("SKILL.md"):
        try:
            content = manifest.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in SKILL_NAMES:
            if f"name: {name}" in content or f'name: "{name}"' in content:
                return manifest

    candidates_text = "\n".join(f"- {candidate}" for candidate in SKILL_MANIFEST_CANDIDATES)
    raise FileNotFoundError(
        "Expected skill manifest not found in cloned skills repo. "
        f"Checked candidates:\n{candidates_text}\n"
        f"Repo: {SKILLS_REPO_URL}"
    )


def copy_skills_repo(destination: Path) -> Path:
    """Clone the skills repo at the pinned commit into a per-run destination."""
    _clone_skills_repo(destination)
    return _find_skill_manifest(destination)


def prepare_skills_directory(
    manifest_path: Path,
    destination: Path,
) -> Path:
    """Create a filtered skills directory containing only the target skill."""
    if destination.exists():
        shutil.rmtree(destination)

    skill_dir = manifest_path.parent
    target_dir = destination / skill_dir.name
    ignore = shutil.ignore_patterns(".git", ".venv", "__pycache__")
    shutil.copytree(skill_dir, target_dir, ignore=ignore)
    return destination


def copy_skill_runtime_assets(skill_dir: Path, workspace: Path) -> None:
    """Copy skill scripts into the workspace so relative paths resolve."""
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return
    target = workspace / "scripts"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(scripts_dir, target)


def _recover_output_file(
    workspace: Path,
    skills_repo_dir: Path,
    output_filename: str,
) -> None:
    target = workspace / output_filename
    if target.exists():
        return

    candidates = list(skills_repo_dir.rglob(output_filename))
    if not candidates:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], target)


def build_fast_agent(environment_dir: Path, skills_manifest_dir: Path) -> FastAgent:
    """Create a fast-agent  instance configured for a single run."""
    fast = FastAgent(
        "fast-agent example",
        ignore_unknown_args=True,
        config_path=str(CONFIG_PATH),
        environment_dir=environment_dir,
        skills_directory=[skills_manifest_dir],
    )

    @fast.agent(
        name="eval_skill",
        servers=["huggingface"],
        instruction=default_instruction,
    )
    async def eval_skill():
        return None

    return fast


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


async def main():
    args = parse_args()
    num_runs = args.runs
    output_filename = args.output_file

    # Create timestamped batch folder for artifacts
    batch_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
    batch_folder = (ROOT_DIR / "runs" / batch_id).resolve()
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

            # Create run subfolder
            run_folder = batch_folder / f"run_{i}"
            run_folder.mkdir(exist_ok=True)
            workspace = (run_folder / "workspace").resolve()
            env_dir = (run_folder / ".fast-agent").resolve()
            skills_repo_dir = (run_folder / "skills_repo").resolve()
            skills_filtered_dir = (run_folder / "skills_filtered").resolve()

            copy_prompt_assets(workspace)
            manifest_path = copy_skills_repo(skills_repo_dir)
            skills_manifest_dir = prepare_skills_directory(manifest_path, skills_filtered_dir)
            copy_skill_runtime_assets(manifest_path.parent, workspace)

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
                "session_id": "",
                "session_history_file": "",
            }

            try:
                fast = build_fast_agent(env_dir, skills_manifest_dir)
                async with fast.run() as agent:

                    eval_agent = agent.eval_skill
                    manager = get_session_manager()
                    session = manager.create_session(
                        metadata={
                            "batch_id": batch_id,
                            "run_number": i,
                            "output_file": output_filename,
                        }
                    )
                    row["session_id"] = session.info.name

                    with run_in_workspace(workspace):
                        # Run the evaluation task
                        prompt_path = Path(PROMPT_SOURCE.name)
                        await agent.interactive()
                        await eval_agent.generate(load_prompt(prompt_path))

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

                    history_path = session.latest_history_path(eval_agent.name)
                    if history_path:
                        row["session_history_file"] = str(history_path)

                _recover_output_file(workspace, skills_repo_dir, output_filename)

                # Validate the output file
                output_path = workspace / output_filename
                validation: ValidationResult = validate_with_metrics(output_path)

                row["passed"] = validation.passed
                row["assertions_passed"] = validation.assertions_passed
                row["assertions_total"] = validation.assertions_total
                row["metrics_count"] = validation.metrics_count
                row["benchmarks_found"] = ",".join(validation.benchmarks_found)
                row["error_message"] = validation.error_message or ""

                status = "PASSED" if validation.passed else "FAILED"
                print(
                    f"Run {i}: {status} ("
                    f"{validation.assertions_passed}/{validation.assertions_total} assertions)"
                )

            except Exception as e:
                row["error_message"] = str(e)
                print(f"Run {i}: ERROR - {e}")

            # Copy any extra artifacts created outside the workspace (rare)
            moved = []
            for pattern in ["*.yaml", "*.py"]:
                for f in Path(".").glob(pattern):
                    if f.name in EXCLUDE_FILES:
                        continue
                    dest = run_folder / f.name
                    shutil.move(str(f), str(dest))
                    moved.append(f.name)
            if moved:
                print(f"Collected artifacts from repo root: {moved}")

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
