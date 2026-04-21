"""Adım 2 — Olasılık dağılımları ve örnekleme dağılımları.

VBM655 konusu: "Olasılık kavramları, olasılık dağılımları, örnekleme dağılımları".
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from utils import RESULTS_DIR, load_dataset, save_fig

RNG = np.random.default_rng(42)


def normality_diagnostics(series: pd.Series, label: str) -> dict:
    sample = series.sample(n=min(5000, len(series)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    ad_res = stats.anderson(series.dropna().to_numpy(), dist="norm")
    skew = float(stats.skew(series))
    kurt = float(stats.kurtosis(series, fisher=True))
    return {
        "feature": label,
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "skew": skew,
        "kurtosis_excess": kurt,
        "shapiro_W": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "anderson_stat": float(ad_res.statistic),
        "anderson_crit_5pct": float(ad_res.critical_values[2]),
    }


def fit_parametric(series: pd.Series) -> dict:
    clipped = series.clip(1e-4, 1 - 1e-4) if series.max() <= 1 else series
    out = {}
    mu, sigma = stats.norm.fit(series)
    out["normal"] = {"mu": float(mu), "sigma": float(sigma)}
    if series.max() <= 1 and series.min() >= 0:
        a, b, _, _ = stats.beta.fit(clipped, floc=0, fscale=1)
        out["beta"] = {"alpha": float(a), "beta": float(b)}
    if (series > 0).all():
        shape, loc, scale = stats.lognorm.fit(series, floc=0)
        out["lognorm"] = {"shape": float(shape), "scale": float(scale)}
    return out


def central_limit_demo(values: np.ndarray) -> dict:
    """Tek tek dağılmayan verinin örnek ortalamalarının normale yaklaşması."""
    n_samples = 2000
    sample_sizes = [5, 30, 200]
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(15, 4.2))
    summary = {}
    for ax, n in zip(axes, sample_sizes):
        means = [values[RNG.integers(0, len(values), size=n)].mean()
                 for _ in range(n_samples)]
        sns.histplot(means, bins=35, ax=ax, color="slateblue", edgecolor="white")
        ax.set_title(f"Örnek Ortalamaları — n={n}")
        ax.set_xlabel("Ortalama popülerlik")
        summary[f"n_{n}"] = {
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means, ddof=1)),
            "theoretical_se": float(values.std(ddof=1) / np.sqrt(n)),
        }
    plt.suptitle("Merkezi Limit Teoremi Gösterimi — Popülerlik")
    save_fig("fig_06_clt_demo.png")
    return summary


def main() -> dict:
    df = load_dataset()
    results: dict = {"normality": {}, "fits": {}}

    for feat in ["popularity", "valence", "energy", "danceability"]:
        results["normality"][feat] = normality_diagnostics(df[feat], feat)
        results["fits"][feat] = fit_parametric(df[feat].dropna())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flatten(),
                        ["popularity", "valence", "energy", "danceability"]):
        stats.probplot(df[feat].sample(5000, random_state=0), dist="norm", plot=ax)
        ax.set_title(f"Q-Q grafiği — {feat}")
    plt.suptitle("Normallik için Q-Q Grafikleri")
    save_fig("fig_07_qq_plots.png")

    plt.figure(figsize=(10, 5))
    xs = np.linspace(0, 1, 400)
    sns.histplot(df["valence"], bins=40, stat="density",
                 color="wheat", edgecolor="white", label="Ampirik")
    beta_fit = results["fits"]["valence"]["beta"]
    plt.plot(xs, stats.beta.pdf(xs, beta_fit["alpha"], beta_fit["beta"]),
             "r-", lw=2, label=f"Beta(α={beta_fit['alpha']:.2f}, β={beta_fit['beta']:.2f})")
    plt.title("Valans için Beta Dağılımı Uyumu")
    plt.xlabel("Valans")
    plt.ylabel("Yoğunluk")
    plt.legend()
    save_fig("fig_08_valence_beta_fit.png")

    results["clt_demo"] = central_limit_demo(df["popularity"].to_numpy())

    with open(RESULTS_DIR / "step_02_distributions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
