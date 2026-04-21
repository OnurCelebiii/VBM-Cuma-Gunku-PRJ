"""Adım 1 — Veri toplama, betimleyici istatistikler ve görselleştirme.

VBM655 konu: "Veri, veri toplama ve betimleme".
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import AUDIO_FEATURES, RESULTS_DIR, load_dataset, save_fig


def main() -> dict:
    df = load_dataset()
    results: dict = {"n_tracks": int(len(df)), "n_genres": int(df["track_genre"].nunique())}

    desc = df[AUDIO_FEATURES + ["popularity", "duration_ms"]].describe().round(4)
    desc.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    results["descriptive_csv"] = "results/descriptive_statistics.csv"

    plt.figure()
    sns.histplot(df["popularity"], bins=40, color="steelblue", edgecolor="white")
    plt.title("Popülerlik Puanının Dağılımı (0–100)")
    plt.xlabel("Popülerlik")
    plt.ylabel("Şarkı Sayısı")
    save_fig("fig_01_popularity_hist.png")

    plt.figure()
    sns.histplot(df["valence"], bins=40, color="darkorange", edgecolor="white")
    plt.title("Valans (Mutluluk) Puanının Dağılımı")
    plt.xlabel("Valans")
    plt.ylabel("Şarkı Sayısı")
    save_fig("fig_02_valence_hist.png")

    top_genres = df["track_genre"].value_counts().head(20)
    plt.figure(figsize=(9, 7))
    sns.barplot(x=top_genres.values, y=top_genres.index, color="teal")
    plt.title("En Fazla Temsil Edilen 20 Tür (Şarkı Sayısı)")
    plt.xlabel("Şarkı Sayısı")
    plt.ylabel("Tür")
    save_fig("fig_03_top_genres.png")

    plt.figure(figsize=(9, 6))
    corr = df[AUDIO_FEATURES + ["popularity"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"})
    plt.title("Ses Özellikleri ve Popülerlik Korelasyon Matrisi")
    save_fig("fig_04_correlation_heatmap.png")

    genre_pop = (df.groupby("track_genre")["popularity"].mean()
                   .sort_values(ascending=False))
    top10 = genre_pop.head(10)
    bottom10 = genre_pop.tail(10).iloc[::-1]
    combined = pd.concat([top10, bottom10])
    plt.figure(figsize=(9, 7))
    colors = ["seagreen"] * 10 + ["indianred"] * 10
    sns.barplot(x=combined.values, y=combined.index, palette=colors)
    plt.title("Ortalama Popülerliğe Göre İlk 10 ve Son 10 Tür")
    plt.xlabel("Ortalama Popülerlik")
    plt.ylabel("Tür")
    save_fig("fig_05_genre_popularity.png")

    results["top_genres_by_popularity"] = genre_pop.head(10).round(2).to_dict()
    results["bottom_genres_by_popularity"] = genre_pop.tail(10).round(2).to_dict()

    with open(RESULTS_DIR / "step_01_eda.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
