"""
AI Resume Screening System — Main Pipeline
Run this file to screen resumes against a job description.

Usage:
    python main.py                          # Demo with sample data
    python main.py --jd path/to/jd.txt     # Custom JD + sample resumes
    python main.py --folder path/to/resumes # Screen folder of resumes
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# Path setup
sys.path.insert(0, str(Path(__file__).parent))

from utils.parser import parse_resume, parse_resume_folder
from models.scorer import ResumeScorer
from data.sample_data import SAMPLE_JD_ML_ENGINEER, SAMPLE_RESUMES


# ─── Visualization ─────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Generate comprehensive scoring visualization."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0F1117')
    for ax in axes.flatten():
        ax.set_facecolor('#1A1D27')
        ax.tick_params(colors='#C8C8D0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E3048')

    COLORS = ['#7C3AED', '#2563EB', '#059669', '#D97706', '#DC2626']
    names  = [n[:15] for n in df['name'].tolist()]

    # ── Chart 1: Final Score Bar ──
    ax = axes[0, 0]
    bars = ax.barh(names[::-1], df['final_score'].tolist()[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(df))][::-1],
                   height=0.6, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Final Score', color='#C8C8D0')
    ax.set_title('🏆 Candidate Rankings', color='#FFFFFF', fontsize=13, pad=12)
    for bar, score in zip(bars, df['final_score'].tolist()[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.2%}', va='center', ha='left', color='#C8C8D0', fontsize=9)

    # ── Chart 2: Score Breakdown Radar-style Bars ──
    ax = axes[0, 1]
    metrics = ['tfidf_similarity', 'skill_match', 'keyword_overlap',
               'education_score', 'experience_score']
    labels  = ['TF-IDF', 'Skills', 'Keywords', 'Education', 'Experience']
    x = np.arange(len(labels))
    width = 0.15
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row['name'][:10],
               color=COLORS[i % len(COLORS)], alpha=0.85)
    ax.set_xticks(x + width * len(df) / 2)
    ax.set_xticklabels(labels, color='#C8C8D0', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('📊 Score Breakdown by Signal', color='#FFFFFF', fontsize=13, pad=12)
    ax.legend(fontsize=7, labelcolor='#C8C8D0', facecolor='#1A1D27', 
              edgecolor='#2E3048', loc='upper right')

    # ── Chart 3: Skill Heatmap ──
    ax = axes[1, 0]
    skill_data = []
    for _, row in df.iterrows():
        skill_data.append([row['tfidf_similarity'], row['skill_match'],
                           row['experience_score'], row['education_score']])
    skill_arr = np.array(skill_data)
    im = ax.imshow(skill_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['TF-IDF', 'Skills', 'Experience', 'Education'],
                       color='#C8C8D0', fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([n[:12] for n in df['name']], color='#C8C8D0', fontsize=9)
    ax.set_title('🔥 Signal Heatmap', color='#FFFFFF', fontsize=13, pad=12)
    for i in range(len(df)):
        for j in range(4):
            ax.text(j, i, f'{skill_arr[i,j]:.2f}', ha='center', va='center',
                    color='#0F1117', fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Chart 4: Percentile Distribution ──
    ax = axes[1, 1]
    scores = df['final_score'].tolist()
    ax.scatter(range(len(scores)), sorted(scores, reverse=True),
               c=[COLORS[i % len(COLORS)] for i in range(len(scores))],
               s=120, zorder=5, edgecolors='white', linewidths=0.5)
    ax.plot(range(len(scores)), sorted(scores, reverse=True),
            color='#6366F1', alpha=0.5, linewidth=1.5)
    threshold = 0.5
    ax.axhline(y=threshold, color='#EF4444', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(len(scores) - 0.5, threshold + 0.01, 'Threshold', color='#EF4444',
            fontsize=8, ha='right')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([n[:10] for n in df.sort_values('final_score', ascending=False)['name']],
                       color='#C8C8D0', fontsize=9, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title('📈 Score Distribution', color='#FFFFFF', fontsize=13, pad=12)
    ax.set_ylabel('Final Score', color='#C8C8D0')

    # Title
    fig.suptitle('AI Resume Screening System — Analysis Report',
                 color='#FFFFFF', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'screening_report.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    plt.close()
    return output_path


# ─── Report Generator ──────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, scorer: ResumeScorer, jd_analysis: dict):
    """Print detailed screening report to console."""
    sep = "═" * 70

    print(f"\n{sep}")
    print(" 🤖  AI RESUME SCREENING SYSTEM — RESULTS")
    print(sep)
    print(f"  Candidates screened : {len(df)}")
    print(f"  Run timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  JD Skills detected  : {', '.join(jd_analysis['skills_flat'][:10])}")
    print(sep)

    print(f"\n{'RANK':<6} {'CANDIDATE':<20} {'SCORE':>8} {'PERCENTILE':>12} {'STATUS':<15}")
    print("─" * 65)

    for _, row in df.iterrows():
        score = row['final_score']
        status = "✅ SHORTLIST" if score >= 0.55 else ("⚠️  MAYBE" if score >= 0.35 else "❌ REJECT")
        print(f"  #{row['rank']:<4} {row['name']:<20} {score:>7.1%} {row['percentile']:>10.0f}th   {status}")

    print(f"\n{sep}")
    print(" 📋  DETAILED BREAKDOWNS")
    print(sep)

    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\n  {'▶'} #{row['rank']} {row['name']} ({row['source']})")
        print(f"     Final Score     : {row['final_score']:.1%}")
        print(f"     TF-IDF Match    : {row['tfidf_similarity']:.1%}")
        print(f"     Skill Match     : {row['skill_match']:.1%}")
        print(f"     Keyword Overlap : {row['keyword_overlap']:.1%}")
        print(f"     Education       : {row['education_score']:.1%}")
        print(f"     Experience      : {row['experience_score']:.1%}  ({row['years_experience']:.0f} yrs)")
        print(f"     Skills Found    : {row['skills_found'][:60]}...")
        print(f"     Word Count      : {row['word_count']}")


def save_csv_report(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    cols = ['rank', 'name', 'source', 'final_score', 'tfidf_similarity',
            'skill_match', 'keyword_overlap', 'education_score',
            'experience_score', 'years_experience', 'skills_found',
            'word_count', 'percentile']
    out = df[cols].copy()
    path = os.path.join(output_dir, 'screening_results.csv')
    out.to_csv(path, index=False)
    return path


# ─── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(jd_text: str, resumes: list[dict]) -> pd.DataFrame:
    """Core pipeline: score resumes and return ranked DataFrame."""
    print("\n  [1/4] Analyzing job description...")
    scorer      = ResumeScorer()
    jd_analysis = scorer.analyze_job_description(jd_text)
    print(f"        Skills extracted: {len(jd_analysis['skills_flat'])}")

    print("  [2/4] Parsing resumes...")
    parsed = []
    for r in resumes:
        if "raw_text" in r:   # already parsed
            parsed.append(r)
        elif "text" in r:     # sample dict with text key
            from utils.parser import parse_resume
            p = parse_resume(r["text"], is_text=True)
            p["name"] = r.get("name", p["name"])
            parsed.append(p)
    print(f"        Parsed {len(parsed)} resumes")

    print("  [3/4] Scoring candidates...")
    df = scorer.score_resumes(jd_text, parsed)

    print("  [4/4] Generating outputs...")
    chart_path = plot_results(df)
    csv_path   = save_csv_report(df)
    print_report(df, scorer, jd_analysis)
    print(f"\n  ✅ Chart saved → {chart_path}")
    print(f"  ✅ CSV saved   → {csv_path}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="AI Resume Screening System")
    parser.add_argument("--jd",     type=str, help="Path to job description .txt file")
    parser.add_argument("--folder", type=str, help="Folder with resume files")
    args = parser.parse_args()

    # Load JD
    if args.jd and os.path.exists(args.jd):
        with open(args.jd) as f:
            jd_text = f.read()
    else:
        print("  Using built-in sample job description (ML Engineer)")
        jd_text = SAMPLE_JD_ML_ENGINEER

    # Load resumes
    if args.folder and os.path.exists(args.folder):
        resumes = parse_resume_folder(args.folder)
    else:
        print("  Using built-in sample resumes (5 candidates)")
        resumes = SAMPLE_RESUMES

    run_pipeline(jd_text, resumes)


if __name__ == "__main__":
    main()
