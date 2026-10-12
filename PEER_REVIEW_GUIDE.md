# Anonymous peer review: structured walkthrough

**Project:** HealthLens AI (Module 03, AI in Healthcare)  
**Course:** ITAI-2372 — Artificial Intel Applications  
**Professor:** Srikanth Bhakthan  
**Student:** José Miguel Tuozzo Izaguirre  
**Time to review:** ~5–8 minutes  
**Reviewer:** [Optional; assignment may use anonymous peer review]

## What the project is

A working NLP **educational** prototype. It identifies similarity between a fictional symptom description and synthetic topic examples, prioritizes explicit warning phrases, and declines unsupported inputs. **It is not a medical symptom checker ready for patient use.**

## How to verify

1. Follow the installation steps in `README.md` and run `python app.py`.
2. Test the fictional itchy-rash scenario; confirm a topic, nearest example, and cosine score appear.
3. Test the fictional chest-pain scenario; confirm the warning layer takes priority over topic matching.
4. Test `No chest pain, but I cannot breathe.`; confirm the warning remains visible.
5. Test a topic-unrelated sentence and `I feel fine and have no symptoms.`; confirm abstention.
6. Run `python -m pytest -q` and `python evaluate.py`.
7. Check the citations, study limitations, and provenance in `PROJECT_REPORT.md` (also available as `PROJECT_REPORT.pdf`).

## Suggested peer-review questions

- Is the healthcare/NLP problem and educational scope stated clearly?
- Does the interface actually run, and can you explain why each output is produced?
- Are the data, features, algorithms, threshold, and example cases transparent?
- Do the tests demonstrate core behavior, including failures/unknowns?
- Does the documentation avoid interpreting synthetic scores as real clinical accuracy?
- Are the ethical issues of safety, bias, uncertainty, and privacy acknowledged?
- Are cited sources attributable and relevant?
- Can another student reproduce the application from the GitHub repository?

## Important cautions when peer reviewing

Please do not test it using actual patient details. This project is intentionally small; missing warning phrases and language mismatches do **not** indicate that a person is well. Do not interpret its class labels or scores as diagnostic or treatment guidance.
