"""Adım 3 — İstatistiksel kestirim ve hipotez testleri.

Ana hipotez: "Mutlu (yüksek valans) şarkılar, üzgün (düşük valans) şarkılardan
daha popülerdir." Yan hipotezler enerji, danslanabilirlik, majör/minör mod ve
explicit sözler için yürütülür.

VBM655 konusu: "İstatistiksel kestirim ve hipotez testleri".
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import RESULTS_DIR, load_dataset, save_fig


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def mean_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    mean = float(x.mean())
    se = float(stats.sem(x))
    h = se * stats.t.ppf(1 - alpha / 2, len(x) - 1)
    return mean, mean - h, mean + h


def diff_mean_ci(a: np.ndarray, b: np.ndarray,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (se ** 4) / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    h = se * stats.t.ppf(1 - alpha / 2, df)
    return float(diff), float(diff - h), float(diff + h)


def two_group_test(df: pd.DataFrame, feature: str,
                   pos_mask: pd.Series, neg_mask: pd.Series,
                   pos_name: str, neg_name: str,
                   hypothesis: str) -> dict:
    a = df.loc[pos_mask, feature].to_numpy()
    b = df.loc[neg_mask, feature].to_numpy()
    welch = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    mw = stats.mannwhitneyu(a, b, alternative="greater")
    diff, lo, hi = diff_mean_ci(a, b)
    return {
        "feature": feature, "pos": pos_name, "neg": neg_name,
        "hypothesis": hypothesis,
        "n_pos": int(len(a)), "n_neg": int(len(b)),
        "mean_pos": float(a.mean()), "mean_neg": float(b.mean()),
        "std_pos": float(a.std(ddof=1)), "std_neg": float(b.std(ddof=1)),
        "mean_diff": diff, "ci95_low": lo, "ci95_high": hi,
        "welch_t": float(welch.statistic), "welch_p_onesided": float(welch.pvalue),
        "mannwhitney_U": float(mw.statistic), "mannwhitney_p_onesided": float(mw.pvalue),
        "cohen_d": cohen_d(a, b),
    }


def main() -> dict:
    df = load_dataset()
    results: dict = {"population_summary": {}, "tests": []}

    mean, lo, hi = mean_ci(df["popularity"].to_numpy())
    results["population_summary"]["popularity"] = {
        "mean": mean, "ci95_low": lo, "ci95_high": hi,
        "n": int(len(df)),
    }
    mean_v, lo_v, hi_v = mean_ci(df["valence"].to_numpy())
    results["population_summary"]["valence"] = {
        "mean": mean_v, "ci95_low": lo_v, "ci95_high": hi_v,
    }

    happy = df["mood"] == "Mutlu"
    sad = df["mood"] == "Üzgün"
    results["tests"].append(two_group_test(
        df, "popularity", happy, sad, "Mutlu", "Üzgün",
        "H1: Mutlu şarkılar üzgün şarkılardan daha popülerdir.",
    ))

    high_e = df["energy"] >= df["energy"].median()
    low_e = df["energy"] < df["energy"].median()
    results["tests"].append(two_group_test(
        df, "popularity", high_e, low_e, "Yüksek enerji", "Düşük enerji",
        "H2: Yüksek enerjili şarkılar daha popülerdir.",
    ))

    high_d = df["danceability"] >= df["danceability"].median()
    low_d = df["danceability"] < df["danceability"].median()
    results["tests"].append(two_group_test(
        df, "popularity", high_d, low_d, "Yüksek danslanabilirlik",
        "Düşük danslanabilirlik",
        "H3: Danslanabilirliği yüksek şarkılar daha popülerdir.",
    ))

    major = df["mode"] == 1
    minor = df["mode"] == 0
    results["tests"].append(two_group_test(
        df, "popularity", major, minor, "Majör", "Minör",
        "H4: Majör tondaki şarkılar minör tondaki şarkılardan daha popülerdir.",
    ))

    explicit = df["explicit"] == True  # noqa: E712
    clean = df["explicit"] == False  # noqa: E712
    results["tests"].append(two_group_test(
        df, "popularity", explicit, clean, "Explicit", "Explicit olmayan",
        "H5: Explicit şarkılar daha popülerdir.",
    ))

    plt.figure()
    sns.boxplot(data=df, x="mood", y="popularity",
                palette=["#4c72b0", "#dd8452", "#55a868"], order=["Üzgün", "Nötr", "Mutlu"])
    plt.title("Ruh Hali Sınıfına Göre Popülerlik")
    plt.xlabel("Ruh hali (Valans'a göre)")
    plt.ylabel("Popülerlik")
    save_fig("fig_09_mood_vs_popularity_box.png")

    plt.figure()
    sns.violinplot(data=df, x="mode", y="popularity", inner="quartile",
                   palette=["#d62728", "#2ca02c"])
    plt.xticks([0, 1], ["Minör (0)", "Majör (1)"])
    plt.title("Majör ve Minör Tondaki Şarkılarda Popülerlik")
    plt.xlabel("Mod")
    plt.ylabel("Popülerlik")
    save_fig("fig_10_mode_vs_popularity.png")

    plt.figure()
    labels = [t["hypothesis"].split(":")[0] for t in results["tests"]]
    diffs = [t["mean_diff"] for t in results["tests"]]
    errs_low = [t["mean_diff"] - t["ci95_low"] for t in results["tests"]]
    errs_high = [t["ci95_high"] - t["mean_diff"] for t in results["tests"]]
    colors = ["green" if d > 0 else "crimson" for d in diffs]
    plt.errorbar(diffs, labels, xerr=[errs_low, errs_high], fmt="o",
                 ecolor="gray", capsize=4, mfc="black", mec="black")
    for i, (d, c) in enumerate(zip(diffs, colors)):
        plt.scatter(d, labels[i], color=c, s=60, zorder=3)
    plt.axvline(0, ls="--", color="black", alpha=0.5)
    plt.xlabel("Ortalamalar arası fark (pozitif grup − negatif grup)")
    plt.title("Hipotez Testleri — %95 Güven Aralıkları")
    save_fig("fig_11_hypothesis_effects.png")

    with open(RESULTS_DIR / "step_03_hypothesis_tests.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
