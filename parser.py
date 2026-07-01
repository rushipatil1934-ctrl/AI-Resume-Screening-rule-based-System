"""
Resume Parser
Handles extraction from text, PDF, and DOCX formats.
"""

import os
import re
from pathlib import Path
from utils.nlp_utils import (
    clean_text, extract_skills, extract_contact_info,
    extract_education, extract_years_experience,
    section_split, get_keyword_frequency
)


def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except ImportError:
        print("pdfplumber not available, skipping PDF")
        return ""


def read_docx_file(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        print("python-docx not available, skipping DOCX")
        return ""


def extract_text(path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        return read_pdf_file(path)
    elif ext == '.docx':
        return read_docx_file(path)
    else:  # .txt or unknown
        return read_text_file(path)


def parse_resume(path_or_text: str, is_text: bool = False) -> dict:
    """
    Full resume parsing pipeline.
    
    Args:
        path_or_text: File path OR raw text string
        is_text:      True if path_or_text is raw text, not a file path
    
    Returns:
        Structured dict with all extracted resume fields
    """
    # 1. Extract raw text
    if is_text:
        raw_text = path_or_text
        source   = "text_input"
    else:
        raw_text = extract_text(path_or_text)
        source   = os.path.basename(path_or_text)

    if not raw_text.strip():
        return {"error": "Could not extract text", "source": source}

    # 2. Clean text
    cleaned = clean_text(raw_text)

    # 3. Section split
    sections = section_split(raw_text)

    # 4. Extract structured fields
    skills       = extract_skills(raw_text)
    contact      = extract_contact_info(raw_text)
    education    = extract_education(raw_text)
    years_exp    = extract_years_experience(raw_text)
    keywords     = get_keyword_frequency(cleaned, top_n=30)

    # 5. Compute flat skill list for easy comparison
    all_skills_flat = [s for skills_list in skills.values() for s in skills_list]

    # 6. Name extraction (heuristic: first non-empty line, title-cased)
    name = "Unknown"
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if line and len(line.split()) <= 5 and line.replace(' ', '').isalpha():
            name = line.title()
            break

    return {
        "source":          source,
        "name":            name,
        "raw_text":        raw_text,
        "cleaned_text":    cleaned,
        "sections":        sections,
        "contact":         contact,
        "education":       education,
        "years_experience": years_exp,
        "skills":          skills,
        "skills_flat":     all_skills_flat,
        "top_keywords":    keywords,
        "word_count":      len(cleaned.split())
    }


def parse_resume_folder(folder: str) -> list[dict]:
    """Parse all resumes in a folder."""
    resumes = []
    supported = {'.txt', '.pdf', '.docx'}
    for f in Path(folder).iterdir():
        if f.suffix.lower() in supported:
            print(f"  Parsing: {f.name}")
            parsed = parse_resume(str(f))
            if "error" not in parsed:
                resumes.append(parsed)
    return resumes
