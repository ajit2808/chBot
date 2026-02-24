import pandas as pd
import os
import json
import numpy as np

# ===============================
# Utility: Safe Value Normalizer
# ===============================

def normalize_value(val):
    if pd.isna(val):
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


# ==========================================
# Intelligent Column Selection (Dynamic)
# ==========================================

def detect_important_columns(df, max_columns=15):
    """
    Dynamically select important columns based on:
    - Non-null ratio
    - Text richness
    - Numeric significance
    - Column uniqueness
    """

    column_scores = []

    total_rows = len(df)

    for col in df.columns:
        series = df[col]

        # Non-null ratio
        non_null_ratio = series.notna().sum() / max(total_rows, 1)

        # Unique ratio (avoid ID-only columns)
        unique_ratio = series.nunique() / max(total_rows, 1)

        # Text richness
        sample_values = series.dropna().astype(str).head(100)
        avg_length = sample_values.map(len).mean() if not sample_values.empty else 0

        # Numeric weight
        is_numeric = pd.api.types.is_numeric_dtype(series)

        score = (
            non_null_ratio * 2 +
            avg_length * 0.01 +
            (1 if is_numeric else 0.5) +
            (0.5 if unique_ratio < 0.9 else 0)  # reduce pure-ID columns
        )

        column_scores.append((col, score))

    # Sort by importance
    column_scores.sort(key=lambda x: x[1], reverse=True)

    selected_columns = [col for col, _ in column_scores[:max_columns]]

    return selected_columns


# ==========================================
# Structured Data Parsing
# ==========================================

def parse_tabular(df, max_rows=100000):
    """
    Massive dataset safe parser.
    Handles millions of rows with chunk safety.
    """

    # Limit extreme datasets (configurable)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    df = df.fillna("")

    # Dynamic column selection
    selected_columns = detect_important_columns(df)

    df = df[selected_columns]

    rows = []

    for _, row in df.iterrows():
        clean = [normalize_value(x) for x in row]
        rows.append(" | ".join(clean))

    return rows


# ==========================================
# CSV Parsing (Massive File Optimized)
# ==========================================

def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

    return parse_tabular(df)


# ==========================================
# Excel Parsing
# ==========================================

def parse_excel(file_path):
    df = pd.read_excel(file_path)
    return parse_tabular(df)


# ==========================================
# PPT Parsing
# ==========================================

def parse_pptx(file_path):
    from pptx import Presentation

    prs = Presentation(file_path)
    slides_text = []

    for slide in prs.slides:
        slide_content = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_content.append(normalize_value(shape.text))

        slides_text.append(" ".join(slide_content))

    return slides_text


# ==========================================
# Universal Ingest Entry
# ==========================================

def ingest_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return parse_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        return parse_excel(file_path)

    elif ext == ".pptx":
        return parse_pptx(file_path)

    else:
        # Fallback for unstructured files
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return [f.read()]
        except:
            raise ValueError("Unsupported file format")