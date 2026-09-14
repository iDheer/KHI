"""Load the raw survey export, encode Likert answers and apply quality filters."""
from pathlib import Path

import numpy as np
import pandas as pd

LIKERT = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
NON_RESPONSE = "No Comments"
THEMES = {"T": "Technology", "E": "Education", "S": "Ethics & Society", "V": "Environment"}

# A respondent must answer at least this share of the 60 statements to be kept.
MIN_ANSWERED_FRAC = 0.75


def load_raw(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the raw answers indexed by respondent id and a code -> statement map."""
    raw = pd.read_csv(path, dtype=str)
    id_col = raw.columns[0]
    raw[id_col] = raw[id_col].astype(int)
    raw = raw.set_index(id_col)
    raw.index.name = "respondent"

    statements = {}
    for col in raw.columns:
        code, text = col.split(".", 1)
        statements[code.strip()] = text.strip()
    raw.columns = list(statements)
    return raw, statements


def encode(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Likert labels to -2..+2; blanks and 'No Comments' become NaN."""
    unknown = set(pd.unique(raw.values.ravel())) - set(LIKERT) - {NON_RESPONSE, np.nan}
    unknown = {u for u in unknown if isinstance(u, str)}
    if unknown:
        raise ValueError(f"Unexpected answer labels: {unknown}")
    return raw.apply(lambda col: col.map(LIKERT)).astype(float)


def quality_report(raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Per-respondent diagnostics used for filtering and documentation."""
    q = pd.DataFrame(index=raw.index)
    q["answered"] = X.notna().sum(axis=1)
    q["blank"] = raw.isna().sum(axis=1)
    q["no_comments"] = (raw == NON_RESPONSE).sum(axis=1)
    q["mean_score"] = X.mean(axis=1)
    q["sd_score"] = X.std(axis=1)
    q["distinct_labels"] = X.nunique(axis=1)
    q["modal_share"] = X.apply(
        lambda r: r.value_counts(normalize=True).max() if r.notna().any() else np.nan, axis=1
    )
    q["kept"] = q["answered"] >= MIN_ANSWERED_FRAC * X.shape[1]
    return q


def theme_of(code: str) -> str:
    return THEMES[code[0]]


def prepare(path: Path):
    raw, statements = load_raw(path)
    X_all = encode(raw)
    quality = quality_report(raw, X_all)
    X = X_all.loc[quality["kept"]].copy()
    return raw, X_all, X, quality, statements
