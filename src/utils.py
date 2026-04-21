"""Ortak yardımcı fonksiyonlar."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "spotify_tracks.csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 5.5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["font.family"] = "DejaVu Sans"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0)
    df = df.dropna(subset=["track_name", "artists"]).copy()
    df = df.drop_duplicates(subset=["track_id"]).copy()
    df["mood"] = pd.cut(
        df["valence"], bins=[-0.01, 0.4, 0.6, 1.01],
        labels=["Üzgün", "Nötr", "Mutlu"],
    )
    df["popularity_class"] = pd.cut(
        df["popularity"], bins=[-1, 25, 50, 75, 101],
        labels=["Düşük", "Orta", "Yüksek", "Hit"],
    )
    df["is_popular"] = (df["popularity"] >= 50).astype(int)
    return df


def save_fig(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out
