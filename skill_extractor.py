import spacy
from spacy.matcher import PhraseMatcher
import subprocess
import sys

class SkillExtractor:
    """
    NLP-based Skill Extractor using spaCy.
    It matches pre-defined tech skills from text.
    """
    def __init__(self, nlp_model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            from spacy.cli import download
            download(nlp_model)
            self.nlp = spacy.load(nlp_model)
        
        # Core IT & Data Skills Database
        skills = [
            "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
            "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
            "sql", "mysql", "postgresql", "mongodb", "aws", "gcp", "azure", "docker",
            "kubernetes", "git", "github", "ci/cd", "machine learning", "deep learning",
            "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn", "pandas",
            "numpy", "spacy", "streamlit", "tf-idf", "cosine similarity", "data analysis",
            "data engineering", "power bi", "tableau", "excel", "agile", "scrum"
        ]
        
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(text) for text in skills]
        self.matcher.add("SKILLS", patterns)

    def extract(self, text: str) -> list[str]:
        """Extracts skills from text using spaCy PhraseMatcher."""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        extracted_skills = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            extracted_skills.add(span.text.lower())
        return list(extracted_skills)
