# Module 03 submission guide

**Student:** José Miguel Tuozzo Izaguirre  
**Course:** ITAI-2372 — Artificial Intel Applications  
**Professor:** Srikanth Bhakthan  
**Repository:** https://github.com/jtuozzo1616/HealthLens-AI-Explainable-Symptom-Analysis

Your course instructions request a recent healthcare AI innovation, credible research, a **working application**, **commented code**, and a **GitHub URL**. Anonymous peer review also applies. This repository addresses the content and coding requirements, but **you must perform the account-specific submission yourself**.

## Before submitting

- Confirm the student, course and professor listed above; check whether your LMS requires a separate section number.
- Run the program locally with `python app.py` and inspect it personally.
- Run `python -m pytest -q` and `python evaluate.py` and keep the output available if needed.
- Explain in your own words that the ML part is TF-IDF + cosine similarity, not a generative LLM, and why that is permitted by Option 2.
- Understand that 100% in a tiny synthetic evaluation is **not medical accuracy**.
- Upload the complete project to your **own** GitHub repository and confirm the URL is visible to graders.
- Review your course's rules on acknowledgment of coding/AI assistance; use accurate disclosures.

## GitHub repository

The public repository has already been created: https://github.com/jtuozzo1616/HealthLens-AI-Explainable-Symptom-Analysis

Upload the *contents* of the `HealthLens_AI` directory to the repository root using GitHub’s web upload interface. Verify the displayed paths (especially `app.py`, `README.md`, `tests/`, and `.github/workflows/python-tests.yml`) and check that automated tests pass.

## Suggested text for the LMS submission field (edit to reflect your work)

**Student:** José Miguel Tuozzo Izaguirre

**Course:** ITAI-2372 — Artificial Intel Applications

**Professor:** Srikanth Bhakthan

**Title:** HealthLens AI — Explainable NLP Application for Educational Symptom-Language Analysis

**GitHub repository:** `https://github.com/jtuozzo1616/HealthLens-AI-Explainable-Symptom-Analysis`

**Description:** This project demonstrates a Python healthcare NLP prototype using TF-IDF vectorization and cosine similarity over a synthetic dataset of fictional symptom descriptions. It provides a Gradio interface, transparent topic matches, a limited explicit warning-phrase layer, negation-handling examples, and abstention for unsupported inputs. The repository includes commented code, a technical/research report, cited sources, a reproducible synthetic evaluation, and automated tests. It is not a diagnosis or clinically validated triage tool.

**How to run:** Install requirements from `requirements.txt`, then run `python app.py`. Execute `python -m pytest -q` and `python evaluate.py` for testing.

**AI assistance disclosure:** ChatGPT assisted with project planning, Python code generation, synthetic examples, testing, and report drafting; the assignment explicitly permits these uses. The application is educational and is not clinically validated. I will review and run the code before making personal verification claims.

## What cannot be completed automatically here

Your LMS submission and any peer-review assignment must be completed through the corresponding account/interface. Verify the actual repository contents before submission.
