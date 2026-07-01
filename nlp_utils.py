"""
NLP Utilities for Resume Screening System
Handles text cleaning, keyword extraction, skill identification
"""

import re
import string
from collections import Counter


# ─── Stop Words (lightweight, no NLTK needed) ────────────────────────────────
STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it","its",
    "itself","they","them","their","theirs","themselves","what","which","who",
    "whom","this","that","these","those","am","is","are","was","were","be","been",
    "being","have","has","had","having","do","does","did","doing","a","an","the",
    "and","but","if","or","because","as","until","while","of","at","by","for",
    "with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when","where","why","how",
    "all","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","s","t","can","will","just","don",
    "should","now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn",
    "doesn","hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan",
    "shouldn","wasn","weren","won","wouldn","also","etc","eg","ie","use","used",
    "using","work","working","worked","help","helped","develop","developed",
    "responsible","responsibilities","ability","strong","excellent","good","team"
}


# ─── Skill Taxonomy ───────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby",
        "php", "swift", "kotlin", "go", "golang", "rust", "scala", "r",
        "matlab", "perl", "bash", "shell", "sql", "nosql", "html", "css"
    ],
    "frameworks_libraries": [
        "django", "flask", "fastapi", "react", "angular", "vue", "nodejs",
        "express", "spring", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "opencv",
        "huggingface", "langchain", "selenium", "pytest", "junit", "bootstrap"
    ],
    "databases": [
        "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
        "cassandra", "sqlite", "oracle", "dynamodb", "firebase", "neo4j",
        "mariadb", "cockroachdb", "snowflake", "bigquery"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "jenkins", "gitlab", "github actions", "terraform", "ansible",
        "circleci", "travis", "helm", "prometheus", "grafana", "nginx",
        "apache", "linux", "unix", "git", "ci/cd", "devops"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "neural network",
        "transformer", "bert", "gpt", "llm", "generative ai", "data science",
        "feature engineering", "model training", "hyperparameter tuning",
        "a/b testing", "statistics", "regression", "classification", "clustering",
        "random forest", "xgboost", "gradient boosting", "svm", "cnn", "rnn",
        "lstm", "attention mechanism", "fine-tuning", "rag"
    ],
    "soft_skills": [
        "leadership", "communication", "problem solving", "critical thinking",
        "teamwork", "collaboration", "agile", "scrum", "project management",
        "time management", "analytical", "creativity", "adaptability",
        "mentoring", "coaching", "presentation", "stakeholder management"
    ],
    "tools": [
        "jira", "confluence", "slack", "tableau", "power bi", "excel",
        "jupyter", "vscode", "intellij", "postman", "swagger", "figma",
        "photoshop", "linux", "windows server", "restful api", "graphql",
        "microservices", "kafka", "rabbitmq", "spark", "hadoop", "airflow"
    ]
}

ALL_SKILLS = {skill for skills in SKILL_TAXONOMY.values() for skill in skills}


# ─── Regex Patterns ───────────────────────────────────────────────────────────
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
URL_PATTERN   = re.compile(r'https?://\S+|www\.\S+')
YEAR_PATTERN  = re.compile(r'\b(19|20)\d{2}\b')

SECTION_HEADERS = {
    "education": ["education", "academic", "qualification", "degree", "university", "college"],
    "experience": ["experience", "employment", "work history", "professional", "career", "job"],
    "skills": ["skills", "technical skills", "competencies", "technologies", "expertise", "proficiencies"],
    "projects": ["projects", "portfolio", "personal projects", "side projects", "open source"],
    "certifications": ["certifications", "certificates", "awards", "achievements", "licenses"],
    "summary": ["summary", "objective", "profile", "about", "overview", "introduction"]
}


# ─── Core Text Functions ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean raw text: lowercase, remove special chars, normalize whitespace."""
    text = text.lower()
    text = URL_PATTERN.sub(' ', text)
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r'\b\w[\w\.\+\#]*\b', text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def extract_ngrams(text: str, n: int = 2) -> list[str]:
    """Extract n-grams from text for multi-word skill matching."""
    words = text.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def extract_skills(text: str) -> dict:
    """
    Extract skills from text using the taxonomy.
    Returns dict with category → list of found skills.
    """
    text_lower = text.lower()
    found = {}
    
    # Check unigrams + bigrams + trigrams
    unigrams = set(text_lower.split())
    bigrams  = set(extract_ngrams(text_lower, 2))
    trigrams = set(extract_ngrams(text_lower, 3))
    all_grams = unigrams | bigrams | trigrams

    for category, skills in SKILL_TAXONOMY.items():
        matched = [s for s in skills if s in all_grams or s in text_lower]
        if matched:
            found[category] = list(set(matched))

    return found


def extract_contact_info(text: str) -> dict:
    """Extract email, phone, URLs from resume text."""
    return {
        "email":  EMAIL_PATTERN.findall(text),
        "phone":  PHONE_PATTERN.findall(text) if PHONE_PATTERN.search(text) else [],
        "urls":   URL_PATTERN.findall(text)
    }


def extract_education(text: str) -> list[str]:
    """Extract education-related lines."""
    degrees = ["phd", "ph.d", "doctorate", "master", "m.s.", "m.sc", "mba",
               "bachelor", "b.s.", "b.sc", "b.tech", "b.e.", "associate",
               "diploma", "b.a.", "m.tech", "m.e.", "b.com", "m.com"]
    lines = text.lower().split('\n')
    return [l.strip() for l in lines if any(d in l for d in degrees)]


def extract_years_experience(text: str) -> float:
    """Estimate years of experience from text patterns."""
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+of\s+experience',
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            return float(m.group(1))
    
    # Fallback: count unique years mentioned (rough proxy)
    years = [int(y) for y in YEAR_PATTERN.findall(text) if 1990 <= int(y) <= 2025]
    if len(years) >= 2:
        return float(max(years) - min(years))
    return 0.0


def get_keyword_frequency(text: str, top_n: int = 20) -> list[tuple]:
    """Return top N keywords with frequency."""
    tokens = tokenize(text)
    return Counter(tokens).most_common(top_n)


def section_split(text: str) -> dict:
    """
    Split resume into sections based on header keywords.
    Returns dict: section_name → content
    """
    lines = text.split('\n')
    sections = {"raw": text}
    current_section = "general"
    buffer = []

    for line in lines:
        line_lower = line.strip().lower()
        matched_section = None
        for section, keywords in SECTION_HEADERS.items():
            if any(kw in line_lower for kw in keywords) and len(line_lower) < 60:
                matched_section = section
                break
        
        if matched_section:
            if buffer:
                sections[current_section] = ' '.join(buffer)
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line.strip())

    if buffer:
        sections[current_section] = ' '.join(buffer)

    return sections


def compute_keyword_overlap(text1: str, text2: str) -> float:
    """Jaccard similarity between keyword sets of two texts."""
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)
