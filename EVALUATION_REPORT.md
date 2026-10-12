# Evaluation report: fixed synthetic benchmark

**Student:** José Miguel Tuozzo Izaguirre  
**Course:** ITAI-2372 — Artificial Intel Applications  
**Professor:** Srikanth Bhakthan  
**Tool:** `python evaluate.py`  
**Scope:** Unit-level functionality for a classroom NLP prototype; **no patients** and **no clinical validation**  
**Validation environment:** Python 3.13, scikit-learn 1.8.0, pytest 9.0.2

## Test data and metrics

- **Training corpus:** 56 fictional sentences; 7 symptom-language categories; 8 sentences each.
- **Evaluation corpus:** 26 hand-written, separate fictional scenarios, pre-labeled with expected topics or phrase warnings.
- **Warning expected-positive scenarios:** 8.
- **Warning expected-negative scenarios:** 18.
- **Topic expected-positive scenarios:** 16.
- **Abstention expected scenarios:** 2.

| Measure | Numerator / denominator | Synthetic benchmark result |
| --- | --- | --- |
| Warning-pattern recall | 8/8 expected-warning scenarios flagged | **100%** |
| Warning-pattern specificity | 18/18 other scenarios not flagged | **100%** |
| Topic matching accuracy | 16/16 expected-topic scenarios correct | **100%** |
| Abstention on designated unknowns | 2/2 cases abstained | **100%** |

**These results are specific to this tiny developer-created benchmark** and have no medically interpretable confidence interval, diagnosis sensitivity, triage sensitivity, or generalization guarantee. The 26 examples are not externally collected, independently clinician-adjudicated, or demographically representative. The labels record the developer's intended behavior, not actual diagnoses. High scores were expected for this deliberately limited educational coverage.

## Specific boundary cases

1. `No chest pain, but I cannot breathe.`: Emergency-warning phrase survives a negated warning in a prior clause.
2. `I deny shortness of breath; my muscles are sore.`: Simple negation is not treated as a positive warning.
3. `I feel fine and have no symptoms.`: The model abstains instead of assigning a symptom topic.
4. `My favorite movie is funny and the weather is nice.`: The model abstains on out-of-domain language.
5. `I have chest pain and a cough.`: Safety warning precedes the otherwise probable respiratory topic.

## Important untested cases

Real patient symptoms, spelling errors, dialects, complex multi-clause negation, pediatric cases, pregnancy, medical history, contradictory statements, indirect symptom descriptions, multilingual text, and adversarial prompts. No actual safety assurance is supported by this evaluation.

## Repeat and audit

```bash
python -m pytest -q
python evaluate.py
```

The evaluation cases are visible and editable in `data/evaluation_cases.json`. To avoid overclaiming, the UI does **not** display synthetic benchmark numbers as patient performance estimates.
