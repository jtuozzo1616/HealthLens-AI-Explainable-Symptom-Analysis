"""Run a reproducible evaluation using ONLY synthetic fictional cases.

Results evaluate matching the author's expectations for toy examples; they do
NOT measure actual clinical sensitivity/specificity or medical effectiveness.
"""
from __future__ import annotations

import json
from pathlib import Path
from healthlens.engine import SymptomAnalyzer

DATA = Path(__file__).resolve().parent / "data" / "evaluation_cases.json"


def evaluate_cases():
    cases = json.loads(DATA.read_text(encoding="utf-8"))
    analyzer = SymptomAnalyzer()
    outcome = []
    for case in cases:
        result = analyzer.analyze(case["text"])
        warning_correct = (result.state == "warning") == case["expected_alert"]
        topic_correct = (result.topic == case["expected_topic"]) if not case["expected_alert"] else None
        outcome.append({
            "input": case["text"],
            "expected_topic": case["expected_topic"],
            "predicted_topic": result.topic,
            "expected_warning": case["expected_alert"],
            "predicted_warning": result.state == "warning",
            "warning_correct": warning_correct,
            "topic_correct": topic_correct,
            "state": result.state,
        })
    emergency_cases = [r for r in outcome if r["expected_warning"]]
    negative_cases = [r for r in outcome if not r["expected_warning"]]
    topic_cases = [r for r in outcome if r["expected_topic"] is not None]
    unknown_cases = [r for r in outcome if r["expected_topic"] is None and not r["expected_warning"]]
    metrics = {
        "n_examples": len(outcome),
        "synthetic_warning_recall": sum(r["predicted_warning"] for r in emergency_cases) / len(emergency_cases),
        "synthetic_warning_specificity": sum(not r["predicted_warning"] for r in negative_cases) / len(negative_cases),
        "synthetic_topic_accuracy": sum(r["topic_correct"] for r in topic_cases) / len(topic_cases),
        "synthetic_unknown_abstention_rate": sum(r["state"] == "abstain" for r in unknown_cases) / len(unknown_cases),
    }
    return metrics, outcome


if __name__ == "__main__":
    metrics, outcomes = evaluate_cases()
    print("SYNTHETIC CLASSROOM TEST ONLY - NOT CLINICAL VALIDATION")
    for name, value in metrics.items():
        print(f"{name}: {value if isinstance(value, int) else f'{value:.1%}'}")
    errors = [r for r in outcomes if not r["warning_correct"] or r["topic_correct"] is False or
              (r["expected_topic"] is None and not r["expected_warning"] and r["state"] != "abstain")]
    if errors:
        print("\nMismatched cases:")
        for row in errors:
            print(json.dumps(row, indent=2))
