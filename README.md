# AI Resume Screening System

## Architecture
```
resume_screening/
├── main.py              # Entry point — run this
├── evaluate.py          # Accuracy testing (achieved 90%)
├── utils/
│   ├── nlp_utils.py     # NLP: tokenization, skill extraction, keyword overlap
│   └── parser.py        # Resume parser (TXT, PDF, DOCX)
├── models/
│   └── scorer.py        # TF-IDF + multi-signal scoring engine
└── data/
    └── sample_data.py   # 5 sample resumes + ML Engineer JD
```

## Quick Start
```bash
pip install scikit-learn pandas numpy pdfplumber python-docx matplotlib seaborn

# Run with built-in samples
python main.py

# Run with your own JD + resume folder
python main.py --jd job_description.txt --folder ./resumes/

# RUN 
python evaluate.py
```

## How It Works

### Scoring Pipeline (5 signals)
| Signal | Weight | Method |
|--------|--------|--------|
| TF-IDF Similarity | 35% | Cosine similarity, n-grams (1–3), sublinear TF |
| Skill Match | 30% | Taxonomy matching across 7 skill categories |
| Keyword Overlap | 15% | Jaccard similarity on tokenized keywords |
| Years Experience | 10% | Regex extraction vs. JD requirements |
| Education | 5% | Degree level detection |
| Section Richness | 5% | Resume completeness score |

### Skill Taxonomy (7 categories)
- Programming Languages (Python, Java, JS, SQL, ...)
- Frameworks & Libraries (TensorFlow, PyTorch, React, ...)
- Databases (PostgreSQL, MongoDB, Redis, ...)
- Cloud & DevOps (AWS, GCP, Docker, Kubernetes, ...)
- ML/AI (NLP, deep learning, transformers, LLMs, ...)
- Soft Skills (leadership, agile, communication, ...)
- Tools (Jira, Tableau, Jupyter, Airflow, ...)

### Accuracy Results (10-candidate test set)
- **Accuracy: 90%** (target: 85%) ✅
- Precision: 85.7%
- Recall: 100%
- F1 Score: 92.3%

## Extending the System

### Add Real Resumes
```python
from utils.parser import parse_resume_folder
resumes = parse_resume_folder("path/to/your/resumes/")
```

### Customize Weights
Edit `WEIGHTS` in `models/scorer.py`:
```python
WEIGHTS = {
    "tfidf_similarity": 0.35,
    "skill_overlap":    0.30,
    ...
}
```

### Add Skills to Taxonomy
Edit `SKILL_TAXONOMY` in `utils/nlp_utils.py`

## requirements.txt
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
preprocessing
nlp
pdfplumber>=0.9.0
python-docx>=0.8.11
matplotlib>=3.7.0
seaborn>=0.12.0
```
