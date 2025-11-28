#!/usr/bin/env python3
"""
Assertions for testing the evaluation extraction output.
Run this after the one-shot prompt completes.
"""

import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Structured result from validation with metrics."""
    passed: bool
    assertions_passed: int
    assertions_total: int
    metrics_count: int
    benchmarks_found: list[str] = field(default_factory=list)
    error_message: str | None = None

def validate_evaluation_file(
    output_file: Path,
    *,
    expected_model: str = "OLMo-7B",
    expected_source: str = "huggingface.co/allenai/OLMo-7B",
    min_expected_benchmarks: int = 9,
    expected_benchmarks: set[str] | None = None,
):
    """Validate an evaluation YAML file and raise on failure."""
    
    assert output_file.exists(), f"Output file '{output_file}' not found"
    print("✓ File exists")
    
    # Load YAML
    with open(output_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Assertion 2: Has model-index structure
    assert 'model-index' in data, "Missing 'model-index' key"
    assert len(data['model-index']) > 0, "model-index is empty"
    print("✓ Valid model-index structure")
    
    model_entry = data['model-index'][0]
    
    # Assertion 3: Model name is correct
    assert model_entry['name'] == expected_model, f"Model name is '{model_entry['name']}', expected '{expected_model}'"
    print("✓ Correct model name")
    
    # Assertion 4: Has results
    assert 'results' in model_entry, "Missing 'results' key"
    assert len(model_entry['results']) > 0, "Results is empty"
    print("✓ Has results")
    
    result = model_entry['results'][0]
    
    # Assertion 5: Task type is text-generation
    assert result['task']['type'] == 'text-generation', f"Task type is {result['task']['type']}"
    print("✓ Correct task type")
    
    # Assertion 6: Has metrics
    assert 'metrics' in result, "Missing 'metrics' key"
    metrics = result['metrics']
    assert len(metrics) > 0, "Metrics list is empty"
    print(f"✓ Has {len(metrics)} metrics")
    
    # Assertion 7: Expected benchmark types
    expected_benchmarks = expected_benchmarks or {
        'arc_challenge', 'arc_easy', 'boolq', 'copa', 'hellaswag',
        'openbookqa', 'piqa', 'sciq', 'winogrande', 'mmlu', 'truthfulqa'
    }
    
    # Extract metric types (normalize to handle variations)
    metric_types = set()
    for m in metrics:
        metric_type = m['type'].lower().replace('_(mc2)', '').replace('_(5_shot_mc)', '').replace(' ', '_')
        # Handle special cases
        if 'truthful' in metric_type:
            metric_type = 'truthfulqa'
        if 'mmlu' in metric_type:
            metric_type = 'mmlu'
        metric_types.add(metric_type)
    
    found_benchmarks = expected_benchmarks & metric_types
    assert len(found_benchmarks) >= min_expected_benchmarks, f"Only found {len(found_benchmarks)} expected benchmarks: {found_benchmarks}"
    print(f"✓ Found {len(found_benchmarks)} expected benchmark types: {sorted(found_benchmarks)}")
    
    # Assertion 8: No unwanted metric types (hyperparameters, architecture)
    unwanted_types = {
        'd_model', 'num_heads', 'num_layers', 'batch_size', 'peak_lr', 
        'warmup_steps', 'weight_decay', 'beta1', 'beta2', 'epsilon',
        'sequence_length', '1b', '7b'
    }
    
    unwanted_found = set()
    for m in metrics:
        metric_type = m['type'].lower()
        for unwanted in unwanted_types:
            if unwanted in metric_type:
                unwanted_found.add(metric_type)
    
    assert len(unwanted_found) == 0, f"Found unwanted metric types: {unwanted_found}"
    print("✓ No hyperparameters or architecture specs included")
    
    # Assertion 9: No random baseline scores
    for m in metrics:
        assert 'random' not in m['name'].lower(), f"Found random baseline: {m['name']}"
    print("✓ No random baseline scores")
    
    # Assertion 10: All metrics have valid values
    for m in metrics:
        assert 'value' in m, f"Metric {m['name']} missing value"
        assert isinstance(m['value'], (int, float)), f"Metric {m['name']} value is not numeric"
        assert m['value'] > 0, f"Metric {m['name']} has invalid value: {m['value']}"
    print("✓ All metrics have valid numeric values")
    
    # Assertion 11: Has source attribution
    assert 'source' in result, "Missing 'source' key"
    assert 'url' in result['source'], "Missing source URL"
    assert expected_source in result['source']['url'], "Incorrect source URL"
    print("✓ Has correct source attribution")
    
    # Summary
    print("\n" + "="*50)
    print(f"ALL ASSERTIONS PASSED ✓")
    print(f"Total metrics: {len(metrics)}")
    print(f"Unique benchmark types: {len(metric_types)}")
    print(f"Expected benchmarks found: {len(found_benchmarks)}/{len(expected_benchmarks)}")
    print("="*50)
    
    return True

def test_olmo_evaluation_output(path: str | Path | None = None):
    """Test the generated evaluation YAML file."""
    output_file = Path(path) if path else Path("olmo_7b_evaluations.yaml")
    return validate_evaluation_file(output_file)


def validate_with_metrics(
    output_file: Path,
    *,
    expected_model: str = "OLMo-7B",
    expected_source: str = "huggingface.co/allenai/OLMo-7B",
    min_expected_benchmarks: int = 9,
    expected_benchmarks: set[str] | None = None,
) -> ValidationResult:
    """Validate an evaluation YAML file and return structured metrics.

    This is a wrapper around validate_evaluation_file that catches exceptions
    and returns a ValidationResult with pass/fail status and assertion counts.
    """
    assertions_passed = 0
    assertions_total = 11  # Total number of assertions in validate_evaluation_file
    metrics_count = 0
    benchmarks_found: list[str] = []

    try:
        # Check file exists (assertion 1)
        if not output_file.exists():
            return ValidationResult(
                passed=False,
                assertions_passed=0,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message=f"Output file '{output_file}' not found",
            )
        assertions_passed += 1

        # Load YAML
        with open(output_file, 'r') as f:
            data = yaml.safe_load(f)

        # Assertion 2: Has model-index structure
        if 'model-index' not in data or len(data['model-index']) == 0:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message="Missing or empty 'model-index' key",
            )
        assertions_passed += 1

        model_entry = data['model-index'][0]

        # Assertion 3: Model name is correct
        if model_entry.get('name') != expected_model:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message=f"Model name is '{model_entry.get('name')}', expected '{expected_model}'",
            )
        assertions_passed += 1

        # Assertion 4: Has results
        if 'results' not in model_entry or len(model_entry['results']) == 0:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message="Missing or empty 'results' key",
            )
        assertions_passed += 1

        result = model_entry['results'][0]

        # Assertion 5: Task type is text-generation
        task_type = result.get('task', {}).get('type')
        if task_type != 'text-generation':
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message=f"Task type is '{task_type}', expected 'text-generation'",
            )
        assertions_passed += 1

        # Assertion 6: Has metrics
        if 'metrics' not in result or len(result['metrics']) == 0:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=0,
                benchmarks_found=[],
                error_message="Missing or empty 'metrics' key",
            )
        metrics = result['metrics']
        metrics_count = len(metrics)
        assertions_passed += 1

        # Assertion 7: Expected benchmark types
        expected_benchmarks = expected_benchmarks or {
            'arc_challenge', 'arc_easy', 'boolq', 'copa', 'hellaswag',
            'openbookqa', 'piqa', 'sciq', 'winogrande', 'mmlu', 'truthfulqa'
        }

        metric_types = set()
        for m in metrics:
            metric_type = m['type'].lower().replace('_(mc2)', '').replace('_(5_shot_mc)', '').replace(' ', '_')
            if 'truthful' in metric_type:
                metric_type = 'truthfulqa'
            if 'mmlu' in metric_type:
                metric_type = 'mmlu'
            metric_types.add(metric_type)

        found_benchmarks = expected_benchmarks & metric_types
        benchmarks_found = sorted(found_benchmarks)

        if len(found_benchmarks) < min_expected_benchmarks:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=metrics_count,
                benchmarks_found=benchmarks_found,
                error_message=f"Only found {len(found_benchmarks)} expected benchmarks: {found_benchmarks}",
            )
        assertions_passed += 1

        # Assertion 8: No unwanted metric types
        unwanted_types = {
            'd_model', 'num_heads', 'num_layers', 'batch_size', 'peak_lr',
            'warmup_steps', 'weight_decay', 'beta1', 'beta2', 'epsilon',
            'sequence_length', '1b', '7b'
        }

        unwanted_found = set()
        for m in metrics:
            metric_type = m['type'].lower()
            for unwanted in unwanted_types:
                if unwanted in metric_type:
                    unwanted_found.add(metric_type)

        if len(unwanted_found) > 0:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=metrics_count,
                benchmarks_found=benchmarks_found,
                error_message=f"Found unwanted metric types: {unwanted_found}",
            )
        assertions_passed += 1

        # Assertion 9: No random baseline scores
        for m in metrics:
            if 'random' in m.get('name', '').lower():
                return ValidationResult(
                    passed=False,
                    assertions_passed=assertions_passed,
                    assertions_total=assertions_total,
                    metrics_count=metrics_count,
                    benchmarks_found=benchmarks_found,
                    error_message=f"Found random baseline: {m['name']}",
                )
        assertions_passed += 1

        # Assertion 10: All metrics have valid values
        for m in metrics:
            if 'value' not in m:
                return ValidationResult(
                    passed=False,
                    assertions_passed=assertions_passed,
                    assertions_total=assertions_total,
                    metrics_count=metrics_count,
                    benchmarks_found=benchmarks_found,
                    error_message=f"Metric {m.get('name')} missing value",
                )
            if not isinstance(m['value'], (int, float)):
                return ValidationResult(
                    passed=False,
                    assertions_passed=assertions_passed,
                    assertions_total=assertions_total,
                    metrics_count=metrics_count,
                    benchmarks_found=benchmarks_found,
                    error_message=f"Metric {m.get('name')} value is not numeric",
                )
            if m['value'] <= 0:
                return ValidationResult(
                    passed=False,
                    assertions_passed=assertions_passed,
                    assertions_total=assertions_total,
                    metrics_count=metrics_count,
                    benchmarks_found=benchmarks_found,
                    error_message=f"Metric {m.get('name')} has invalid value: {m['value']}",
                )
        assertions_passed += 1

        # Assertion 11: Has source attribution
        if 'source' not in result or 'url' not in result.get('source', {}):
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=metrics_count,
                benchmarks_found=benchmarks_found,
                error_message="Missing source or source URL",
            )
        if expected_source not in result['source']['url']:
            return ValidationResult(
                passed=False,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                metrics_count=metrics_count,
                benchmarks_found=benchmarks_found,
                error_message=f"Incorrect source URL: {result['source']['url']}",
            )
        assertions_passed += 1

        # All assertions passed
        return ValidationResult(
            passed=True,
            assertions_passed=assertions_passed,
            assertions_total=assertions_total,
            metrics_count=metrics_count,
            benchmarks_found=benchmarks_found,
            error_message=None,
        )

    except Exception as e:
        return ValidationResult(
            passed=False,
            assertions_passed=assertions_passed,
            assertions_total=assertions_total,
            metrics_count=metrics_count,
            benchmarks_found=benchmarks_found,
            error_message=str(e),
        )


if __name__ == "__main__":
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("olmo_7b_evaluations.yaml")
    try:
        validate_evaluation_file(target_file)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
