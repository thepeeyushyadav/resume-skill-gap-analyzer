from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skill_extractor import SkillExtractor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Data Containers
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Holds all output of one JD ↔ Resume comparison."""
    resume_name:       str
    similarity_score:  float                     # 0.0 – 1.0
    matched_skills:    List[str] = field(default_factory=list)
    missing_skills:    List[str] = field(default_factory=list)
    extra_skills:      List[str] = field(default_factory=list)   # resume has but JD doesn't mention
    top_tfidf_terms:   List[Tuple[str, float]] = field(default_factory=list)
    match_category:    str = "Unknown"           # Excellent / Good / Fair / Poor

    def to_dict(self) -> Dict:
        return {
            "Resume":          self.resume_name,
            "Match Score (%)": round(self.similarity_score * 100, 2),
            "Category":        self.match_category,
            "Matched Skills":  ", ".join(self.matched_skills) if self.matched_skills else "None",
            "Missing Skills":  ", ".join(self.missing_skills) if self.missing_skills else "None",
            "Bonus Skills":    ", ".join(self.extra_skills) if self.extra_skills else "None",
        }


# ─────────────────────────────────────────────
#  Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw text:
      • lowercase
      • collapse whitespace
      • keep alphanumerics + spaces (preserve skill tokens like 'c++', 'node.js')
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
#  TF-IDF Engine
# ─────────────────────────────────────────────

class TFIDFEngine:
    """Wraps sklearn TfidfVectorizer with helpers for skill-gap analysis."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        min_df: int = 1,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,        # log(1+tf)
            stop_words="english",
        )
        self._fitted = False

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """Fit on corpus and return TF-IDF matrix (n_docs × n_features)."""
        cleaned = [clean_text(doc) for doc in corpus]
        matrix = self.vectorizer.fit_transform(cleaned)
        self._fitted = True
        return matrix.toarray()

    @staticmethod
    def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        return float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])

    def top_terms(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """Return top-n (term, tfidf_score) pairs for a document vector."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        feature_names = self.vectorizer.get_feature_names_out()
        indices = np.argsort(vector)[::-1][:n]
        return [(feature_names[i], round(float(vector[i]), 4)) for i in indices if vector[i] > 0]


# ─────────────────────────────────────────────
#  Category Helper
# ─────────────────────────────────────────────

def _categorise(score: float) -> str:
    if score >= 0.75:
        return "🏆 Excellent"
    elif score >= 0.55:
        return "✅ Good"
    elif score >= 0.35:
        return "⚠️  Fair"
    else:
        return "❌ Poor"


# ─────────────────────────────────────────────
#  Main Analyser
# ─────────────────────────────────────────────

class ResumeAnalyzer:
    """End-to-end pipeline."""

    def __init__(self, skill_extractor: SkillExtractor = None):
        self.engine = TFIDFEngine()
        self.extractor = skill_extractor or SkillExtractor()

    def analyse(
        self,
        jd_text:      str,
        resumes:      Dict[str, str],   # {filename: raw_text}
    ) -> List[AnalysisResult]:
        if not resumes:
            raise ValueError("Provide at least one resume.")

        names    = list(resumes.keys())
        texts    = [jd_text] + [resumes[n] for n in names]
        matrix   = self.engine.fit_transform(texts)

        jd_vec   = matrix[0]
        res_vecs = matrix[1:]

        jd_skills  = set(self.extractor.extract(jd_text))

        results: List[AnalysisResult] = []
        for idx, name in enumerate(names):
            res_text   = resumes[name]
            res_vec    = res_vecs[idx]
            score      = self.engine.cosine_sim(jd_vec, res_vec)
            res_skills = set(self.extractor.extract(res_text))

            matched = sorted(jd_skills & res_skills)
            missing = sorted(jd_skills - res_skills)
            extra   = sorted(res_skills - jd_skills)

            results.append(AnalysisResult(
                resume_name      = name,
                similarity_score = score,
                matched_skills   = matched,
                missing_skills   = missing,
                extra_skills     = extra,
                top_tfidf_terms  = self.engine.top_terms(res_vec),
                match_category   = _categorise(score),
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def summary_dataframe(self, results: List[AnalysisResult]) -> pd.DataFrame:
        """Convert results list → tidy DataFrame for display / export."""
        return pd.DataFrame([r.to_dict() for r in results])
