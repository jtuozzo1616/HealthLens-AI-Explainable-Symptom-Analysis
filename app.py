"""HealthLens AI interactive web interface. Start with: python app.py

The interface is intended for fictional classroom vignettes, not patient care.
No user text is persisted by the application code; the deployment platform
may have its own access logs and must be reviewed separately.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import gradio as gr

from healthlens.engine import SymptomAnalyzer

# Model is initialized once at startup, not re-trained on user submissions.
ANALYZER = SymptomAnalyzer()

SAMPLES = [
    ["My nose is blocked and I have a cough and a scratchy throat."],
    ["I have an itchy red rash on my hand."],
    ["It burns when I pee and I need to urinate often."],
    ["I have chest pain and trouble breathing."],
    ["No chest pain, but I have a sore back after working out."],
    ["I enjoyed the music at my birthday party."],
]

CSS = """
.gradio-container {max-width: 1070px !important; margin: auto !important;}
.hl-hero {border-radius: 22px; padding: 30px 34px; background: linear-gradient(120deg,#0c2340,#176a85); color: #fff; margin: 8px 0 20px;}
.hl-eyebrow {letter-spacing:.15em; text-transform:uppercase; font-weight:650; font-size:12px; color:#b8f0f3;}
.hl-hero h1 {font-size:39px !important; color: white !important; margin:6px 0 !important; line-height:1.14;}
.hl-hero p {color:#e0eef4; font-size:16px; max-width:740px; margin:4px 0;}
.hl-pill {display:inline-block; border-radius: 25px; padding:6px 12px; background:#e9f1f6; color:#204f67; font-size:12px; font-weight:750; margin-bottom:11px;}
.hl-result {border:1px solid #d9e5ee; border-radius:17px; padding:22px; background: #fff; margin:8px 0;}
.hl-result h2 {font-size:25px; margin:5px 0 9px; color:#13324d;}
.hl-result p {line-height:1.6; color:#304c62;}
.hl-warning {background:#fff6f0; border-color:#eca874;}
.hl-warning .hl-pill {background:#fae0d0; color:#913e20;}
.hl-neutral {background:#f3f7f9;}
.hl-muted {font-size:13px; color:#526578;}
.hl-footer {font-size:12px; color:#5b6975; line-height:1.6;}
"""


def run_analysis(text: str):
    """Render only escaped generated content, never unescaped user HTML."""
    result = ANALYZER.analyze(text)
    if result.state == "warning":
        signs = ", ".join(escape(s) for s in result.warning_signs)
        panel = (
            '<div class="hl-result hl-warning">'
            '<span class="hl-pill">Safety-first pattern match</span>'
            f'<h2>{escape(result.headline)}</h2>'
            f'<p><strong>Detected language:</strong> {signs}</p>'
            f'<p>{escape(result.detail)}</p></div>'
        )
        reasoning = (
            "### Why this result appeared\n"
            "An explicit warning phrase was found by a conservative language-rule layer. "
            "The machine-learning topic classifier was deliberately bypassed. "
            "A missing warning match would **not** mean that symptoms are safe."
        )
    elif result.state == "topic":
        panel = (
            '<div class="hl-result">'
            '<span class="hl-pill">NLP topic similarity</span>'
            f'<h2>{escape(result.headline)}</h2>'
            f'<p>{escape(result.detail)}</p></div>'
        )
        reasoning = (
            "### Explainable NLP result\n"
            f"**Text topic:** `{result.topic}`  \n"
            f"**Nearest synthetic training sentence:** {result.matched_example}  \n"
            f"**Shared vocabulary:** {', '.join(result.overlaps) or 'None'}  \n"
            f"**Cosine similarity:** `{result.similarity:.3f}`  \n\n"
            "*This score measures similarity to an example. It is **not** "
            "a confidence percentage, diagnosis likelihood, or medical risk estimate.*"
        )
    else:
        panel = (
            '<div class="hl-result hl-neutral">'
            '<span class="hl-pill">Model abstention</span>'
            f'<h2>{escape(result.headline)}</h2>'
            f'<p>{escape(result.detail)}</p></div>'
        )
        reasoning = (
            "### Why the model abstained\n"
            "Input was missing, too long, or too different from the limited "
            "synthetic training data. An uncertain result is never a safety clearance."
        )
    return panel, reasoning


def build_demo():
    with gr.Blocks(title="HealthLens AI | Educational NLP", analytics_enabled=False) as demo:
        gr.HTML("""
        <header class="hl-hero">
          <div class="hl-eyebrow">Module 03 · AI in Healthcare</div>
          <h1>HealthLens AI</h1>
          <p>Explainable symptom-language analysis using TF-IDF, cosine similarity,
          and an explicit safety-rule layer.</p>
        </header>
        """)
        gr.Markdown(
            "⚠️ **CLASSROOM PROTOTYPE ONLY.** Use **fictional examples**, not real patient "
            "information. This tool does not diagnose, assess medical urgency comprehensively, "
            "or replace a health professional. For an emergency, call your local emergency number."
        )
        with gr.Row():
            with gr.Column(scale=5):
                text = gr.Textbox(
                    label="Fictional symptom scenario (English)",
                    lines=5, max_lines=8,
                    placeholder="Example: My nose is stuffy, I have a cough, and my throat feels scratchy.",
                    max_length=1000,
                    info="Do not enter names, dates of birth, or real medical information.",
                )
                with gr.Row():
                    analyze = gr.Button("Analyze fictional scenario", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=5):
                result_box = gr.HTML(
                    '<div class="hl-result hl-neutral"><span class="hl-pill">Ready</span>'
                    '<h2>Try a fictional case</h2><p>The system will show its topic or ' 
                    'a safety warning, along with an explanation.</p></div>',
                    label="Analysis result",
                )
        explanation = gr.Markdown("### How it works\nThe model will show an explanation after you analyze a scenario.")
        analyze.click(fn=run_analysis, inputs=text, outputs=[result_box, explanation])
        text.submit(fn=run_analysis, inputs=text, outputs=[result_box, explanation])
        # Clearing resets the displayed interpretation too, avoiding stale outputs.
        clear.click(
            fn=lambda: (
                "",
                '<div class="hl-result hl-neutral"><span class="hl-pill">Ready</span>'
                '<h2>Try a fictional case</h2><p>Results will appear here.</p></div>',
                "### How it works\nThe model will show an explanation after you analyze a scenario.",
            ),
            inputs=None,
            outputs=[text, result_box, explanation],
        )
        gr.Markdown("### Try one of the synthetic demonstration scenarios")
        gr.Examples(examples=SAMPLES, inputs=text, label="Click a scenario to populate the input")
        with gr.Accordion("Methods, safeguards, and source links", open=False):
            gr.Markdown(
                "**Pipeline:** text normalization → explicit warning-pattern check (with a limited "
                "negation heuristic) → TF-IDF unigram/bigram vectorization → cosine similarity to "
                "56 fictional examples → thresholded topic or abstention.\n\n"
                "**Limits:** English only; keyword negation is imperfect; no validated clinical "
                "accuracy, diagnoses, medical risk scores, or medical recommendations; cases are "
                "artificially written. This tool cannot detect all emergencies.\n\n"
                "**Privacy:** The application's code neither saves entered text nor sends it to "
                "a model API. Hosting platforms can maintain network or request logs.\n\n"
                "**Background:** [NIH MedlinePlus: medical emergencies]"
                "(https://medlineplus.gov/ency/article/001927.htm) · "
                "[JMIR 2023 systematic review](https://pubmed.ncbi.nlm.nih.gov/37266983/) · "
                "[WHO AI ethics guidance](https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models) · "
                "[FDA Clinical Decision Support guidance (2026)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)"
            )
        gr.HTML('<footer class="hl-footer">José Miguel Tuozzo Izaguirre • '
                'ITAI-2372 • Professor Srikanth Bhakthan<br>'
                'Educational software • Version 1.0 • Not validated '
                'or authorized for clinical use. For classroom demonstration only.</footer>')
    return demo


def main():
    # Local-only binding avoids an accidental public exposure of demo inputs.
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CSS,
                show_error=False, inbrowser=True)


if __name__ == "__main__":
    main()
