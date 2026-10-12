"""NLP topic retrieval with a conservative, rule-based warning layer.

This classroom prototype NEVER makes a medical diagnosis or clears a user as safe.
The TF-IDF retrieval score describes lexical similarity, not disease probability.
Training and evaluation examples are fictional and are NOT clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_examples.json"
MIN_SCORE = 0.20
MAX_INPUT_CHARS = 1000

TOPIC_DESCRIPTIONS = {
    "respiratory": ("Cough, throat, or nasal symptoms", "A general symptom-language category; not a diagnosis."),
    "headache": ("Head pain or pressure", "A general symptom-language category; not a diagnosis."),
    "digestive": ("Stomach or digestive symptoms", "A general symptom-language category; not a diagnosis."),
    "skin": ("Skin irritation or itching", "A general symptom-language category; not a diagnosis."),
    "musculoskeletal": ("Muscle or joint discomfort", "A general symptom-language category; not a diagnosis."),
    "urinary": ("Urination-related symptoms", "A general symptom-language category; not a diagnosis."),
    "eye_nose_allergy": ("Eye and nose irritation", "Describes symptom language only; does not establish an allergy."),
}

# A small, intentionally conservative set of fictional emergency-warning patterns.
# Not a comprehensive clinical protocol: missing a match NEVER means "not urgent".
WARNING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:chest pain|chest pressure|chest tightness)\b", "Chest discomfort"),
    (r"\b(?:difficulty breathing|trouble breathing|shortness of breath|can(?:not|'t) breathe|struggling to breathe|unable to breathe|not breathing)\b", "Breathing difficulty"),
    (r"\b(?:coughing up blood|cough up blood|vomiting blood|throwing up blood)\b", "Coughing/vomiting blood"),
    (r"\b(?:slurred speech|speech is slurred|face drooping|facial droop|one.sided weakness|cannot speak|can't speak|unable to speak)\b", "Possible neurological warning"),
    (r"\b(?:severe abdominal pain|severe belly pain|sudden severe abdominal pain)\b", "Severe abdominal pain"),
    (r"\b(?:bleeding won't stop|bleeding will not stop|cannot stop bleeding|can't stop bleeding|uncontrolled bleeding)\b", "Uncontrolled bleeding"),
    (r"\b(?:lost consciousness|unconscious|passed out|fainted|fainting)\b", "Loss of consciousness"),
    (r"\b(?:throat is swelling|swollen tongue|tongue swelling|face swelling and trouble breathing)\b", "Potentially serious swelling"),
    (r"\b(?:sudden severe headache|worst headache of my life)\b", "Sudden severe head pain"),
    (r"\b(?:seizure|convulsion)\b", "Possible seizure"),
    (r"\b(?:suicidal thoughts|want to kill myself|thinking of suicide|want to hurt myself)\b", "Self-harm warning"),
)

# Negation cues only suppress a nearby warning phrase within the same short clause.
# This is deliberately a LIMITED English heuristic, not clinical assertion detection.
NEGATION_CUES = {"no", "not", "without", "deny", "denies", "denied", "don't", "dont", "do not", "doesn't", "doesnt", "never", "haven't", "have not"}
CLAUSE_BREAK = re.compile(r"\b(?:but|however|although|yet)\b|[.;!?]", re.I)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)


@dataclass(frozen=True)
class Analysis:
    state: str  # warning, topic, or abstain
    headline: str
    detail: str
    topic: str | None
    similarity: float | None
    matched_example: str | None
    overlaps: tuple[str, ...]
    warning_signs: tuple[str, ...]


def normalize_text(text: str) -> str:
    """Unicode-normalize input and trim repeat whitespace; does not store it."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def _is_negated(clause: str, phrase_start: int) -> bool:
    """Find immediate negation cues up to four tokens before a matched phrase.

    Negation is hard in actual clinical notes (e.g., hypothetical and historical
    mention); keep the heuristic transparent instead of claiming full NLP understanding.
    """
    prefix = clause[:phrase_start].lower()
    nearby = TOKEN_RE.findall(prefix)[-5:]
    if not nearby:
        return False
    joined = " ".join(nearby)
    return any(re.search(rf"(?:^|\s){re.escape(cue)}(?:\s|$)", joined) for cue in NEGATION_CUES)


def find_warnings(text: str) -> tuple[str, ...]:
    """Extract explicit non-negated warning phrases; evaluated BEFORE the ML layer."""
    found: list[str] = []
    for clause in CLAUSE_BREAK.split(normalize_text(text).lower()):
        for expression, label in WARNING_PATTERNS:
            for match in re.finditer(expression, clause):
                if not _is_negated(clause, match.start()) and label not in found:
                    found.append(label)
    return tuple(found)


def _remove_negated_warning_mentions(text: str) -> str:
    """Keep a negated warning phrase from distorting downstream topic similarity.

    This is NOT a medical negation resolver; it only cleans explicit phrases
    already recognized by the small pedagogical warning vocabulary.
    """
    edited = normalize_text(text)
    for expression, _label in WARNING_PATTERNS:
        matches = list(re.finditer(expression, edited, flags=re.I))
        for match in reversed(matches):
            clause_prefix = CLAUSE_BREAK.split(edited[:match.start()])[-1]
            if _is_negated(clause_prefix, len(clause_prefix)):
                edited = edited[:match.start()] + " " + edited[match.end():]
    return normalize_text(edited)


class SymptomAnalyzer:
    """Vector-space NLP demonstration over author-created fictional examples."""

    def __init__(self, data_path: Path = DATA_PATH):
        examples_by_topic = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.examples: list[str] = []
        self.labels: list[str] = []
        for label, samples in examples_by_topic.items():
            if label not in TOPIC_DESCRIPTIONS or not samples:
                raise ValueError(f"Invalid or empty example category: {label}")
            for sample in samples:
                self.examples.append(sample)
                self.labels.append(label)
        # TF-IDF automatically learns term importance FROM synthetic training text.
        # Stop-word removal + 1/2-word n-grams are standard feature engineering.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=2500,
        )
        self.example_vectors = self.vectorizer.fit_transform(self.examples)

    def analyze(self, raw_text: str) -> Analysis:
        text = normalize_text(raw_text)
        if not text:
            return self._abstain("Enter a fictional symptom description to begin.")
        if len(text) > MAX_INPUT_CHARS:
            return self._abstain(f"Please use {MAX_INPUT_CHARS} characters or fewer.")

        # Safety-first ordering prevents a high ML score masking explicit warnings.
        warnings = find_warnings(text)
        if warnings:
            special_crisis = "Self-harm warning" in warnings
            detail = (
                "This classroom example includes language that may describe a serious warning sign. "
                "For an actual emergency in the United States, call 911; elsewhere use your local emergency number. "
                "Do not rely on this prototype to decide what care is needed."
            )
            if special_crisis:
                detail += " In the US, call or text 988 for crisis support; for immediate danger call 911."
            return Analysis(
                state="warning", headline="Warning language detected", detail=detail,
                topic=None, similarity=None, matched_example=None,
                overlaps=(), warning_signs=warnings,
            )

        # Explicit asymptomatic statements should not be assigned to a disease topic.
        # This is an input-scope guard, not evidence that a person is healthy.
        if re.search(r"\b(?:no symptoms|asymptomatic|feel fine|feeling fine|nothing wrong)\b", text, re.I):
            return self._abstain(
                "No active symptom description was identified. This is not a health assessment."
            )

        topic_text = _remove_negated_warning_mentions(text)
        query_vector = self.vectorizer.transform([topic_text])
        scores = cosine_similarity(query_vector, self.example_vectors)[0]
        nearest_index = int(scores.argmax())
        nearest_score = float(scores[nearest_index])
        if nearest_score < MIN_SCORE:
            return self._abstain(
                "The description is outside this small vocabulary, or there is not enough information. "
                "No safety conclusion can be drawn from this result."
            )
        topic = self.labels[nearest_index]
        features = self.vectorizer.get_feature_names_out()
        query_terms = {features[i] for i in query_vector.indices if " " not in features[i]}
        sample_terms = {features[i] for i in self.example_vectors[nearest_index].indices if " " not in features[i]}
        overlaps = tuple(sorted(query_terms & sample_terms))[:8]

        return Analysis(
            state="topic",
            headline=TOPIC_DESCRIPTIONS[topic][0],
            detail=(
                "This is a text-similarity topic, NOT a diagnosis or triage clearance. "
                "A clinician should evaluate real symptoms, especially new, severe, or worsening ones."
            ),
            topic=topic, similarity=nearest_score,
            matched_example=self.examples[nearest_index],
            overlaps=overlaps, warning_signs=(),
        )

    @staticmethod
    def _abstain(message: str) -> Analysis:
        return Analysis(
            state="abstain", headline="Unable to classify reliably", detail=message,
            topic=None, similarity=None, matched_example=None, overlaps=(), warning_signs=(),
        )
