"""Adım 4 — Sınıflanmış sayım verilerinin analizi.

VBM655 konusu: "Sınıflanmış sayım verilerinin analizi" (Kontenjans tabloları,
Ki-kare bağımsızlık testi, Cramér's V, tek yönlü ANOVA).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import RESULTS_DIR, load_dataset, save_fig


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def main() -> dict:
    df = load_dataset()
    results: dict = {}

    mood_pop = pd.crosstab(df["mood"], df["popularity_class"])
    chi2, p, dof, expected = stats.chi2_contingency(mood_pop)
    results["mood_vs_popularity"] = {
        "contingency_table": mood_pop.to_dict(),
        "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "cramers_v": cramers_v(mood_pop),
        "hypothesis": "H0: Ruh hali (valans) ile popülerlik sınıfı bağımsızdır.",
    }

    plt.figure(figsize=(9, 5.5))
    sns.heatmap(mood_pop, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Ruh Hali × Popülerlik Sınıfı Kontenjans Tablosu")
    plt.xlabel("Popülerlik sınıfı")
    plt.ylabel("Ruh hali")
    save_fig("fig_12_contingency_mood_popularity.png")

    mode_pop = pd.crosstab(df["mode"].map({0: "Minör", 1: "Majör"}),
                           df["popularity_class"])
    chi2, p, dof, expected = stats.chi2_contingency(mode_pop)
    results["mode_vs_popularity"] = {
        "contingency_table": mode_pop.to_dict(),
        "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "cramers_v": cramers_v(mode_pop),
    }

    df["loud_class"] = pd.cut(df["loudness"], bins=[-60, -15, -10, -5, 5],
                              labels=["Çok sessiz", "Sessiz", "Yüksek", "Çok yüksek"])
    loud_pop = pd.crosstab(df["loud_class"], df["popularity_class"])
    chi2, p, dof, _ = stats.chi2_contingency(loud_pop)
    results["loudness_vs_popularity"] = {
        "contingency_table": loud_pop.to_dict(),
        "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "cramers_v": cramers_v(loud_pop),
    }

    plt.figure(figsize=(9, 5.5))
    sns.heatmap(loud_pop, annot=True, fmt="d", cmap="Purples")
    plt.title("Ses Şiddeti Sınıfı × Popülerlik Kontenjans Tablosu")
    plt.xlabel("Popülerlik sınıfı")
    plt.ylabel("Ses şiddeti")
    save_fig("fig_13_contingency_loudness.png")

    top10_genres = (df.groupby("track_genre")["popularity"].mean()
                      .sort_values(ascending=False).head(10).index.tolist())
    sub = df[df["track_genre"].isin(top10_genres)]
    groups = [sub.loc[sub["track_genre"] == g, "popularity"].to_numpy()
              for g in top10_genres]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    results["anova_top10_genres"] = {
        "genres": top10_genres,
        "F": float(anova.statistic), "p_value": float(anova.pvalue),
        "kruskal_H": float(kruskal.statistic),
        "kruskal_p": float(kruskal.pvalue),
    }

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sub, x="track_genre", y="popularity",
                order=top10_genres, palette="viridis")
    plt.xticks(rotation=35, ha="right")
    plt.title("En Popüler 10 Tür — Popülerlik Dağılımları (ANOVA)")
    plt.xlabel("Tür")
    plt.ylabel("Popülerlik")
    save_fig("fig_14_anova_top_genres.png")

    with open(RESULTS_DIR / "step_04_categorical.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
