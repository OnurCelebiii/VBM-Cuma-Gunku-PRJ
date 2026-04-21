"""Adım 5 — Çoklu doğrusal regresyon ve lojistik regresyon.

VBM655 konusu: "Regresyon analizi". Ek olarak sınıflandırma amacıyla lojistik
regresyon, k-NN ve Random Forest modelleri karşılaştırılır.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import AUDIO_FEATURES, RESULTS_DIR, load_dataset, save_fig


def linear_regression_analysis(df: pd.DataFrame) -> dict:
    X = df[AUDIO_FEATURES].copy()
    X = sm.add_constant(X)
    y = df["popularity"].astype(float)
    model = sm.OLS(y, X).fit()

    coeffs = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    }).round(5)
    coeffs.to_csv(RESULTS_DIR / "regression_coefficients.csv")

    plt.figure(figsize=(9, 6))
    sig = coeffs.drop("const").sort_values("coef")
    colors = ["green" if c > 0 else "crimson" for c in sig["coef"]]
    plt.barh(sig.index, sig["coef"], color=colors, edgecolor="black")
    plt.axvline(0, color="black", lw=0.7)
    plt.title("OLS Regresyon Katsayıları — Popülerliği Açıklayan Özellikler")
    plt.xlabel("Katsayı (β)")
    save_fig("fig_15_ols_coefficients.png")

    plt.figure()
    sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.15, s=8)
    plt.axhline(0, color="red", ls="--")
    plt.xlabel("Tahmin edilen popülerlik")
    plt.ylabel("Artık (residual)")
    plt.title("OLS Artıkları — Homoskedastisite Kontrolü")
    save_fig("fig_16_ols_residuals.png")

    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "n_obs": int(model.nobs),
        "coefficients": coeffs.to_dict(orient="index"),
        "aic": float(model.aic), "bic": float(model.bic),
    }


def classification_pipeline(df: pd.DataFrame) -> dict:
    X = df[AUDIO_FEATURES].to_numpy()
    y = df["is_popular"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    models = {
        "Lojistik Regresyon": Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=500))]),
        "k-NN (k=25)": Pipeline([("scaler", StandardScaler()),
                                 ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1))]),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                random_state=42, max_depth=18),
    }
    out = {}
    plt.figure(figsize=(7.5, 6))
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        report = classification_report(y_test, pred, output_dict=True,
                                       zero_division=0)
        cm = confusion_matrix(y_test, pred)
        out[name] = {
            "auc": float(auc),
            "accuracy": float(report["accuracy"]),
            "precision_popular": float(report["1"]["precision"]),
            "recall_popular": float(report["1"]["recall"]),
            "f1_popular": float(report["1"]["f1-score"]),
            "confusion_matrix": cm.tolist(),
        }
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrileri — Popüler (≥50) Sınıflandırması")
    plt.legend()
    save_fig("fig_17_roc_curves.png")

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=AUDIO_FEATURES).sort_values()
    plt.figure(figsize=(8, 5.5))
    sns.barplot(x=imp.values, y=imp.index, palette="cividis")
    plt.title("Random Forest — Özellik Önem Skorları")
    plt.xlabel("Önem (Gini)")
    save_fig("fig_18_rf_feature_importance.png")
    out["random_forest_feature_importance"] = imp.round(4).to_dict()
    return out


def logistic_valence_popularity(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df[["valence", "energy", "danceability", "loudness"]])
    y = df["is_popular"]
    model = sm.Logit(y, X).fit(disp=False)
    odds = np.exp(model.params)
    return {
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "params": model.params.round(4).to_dict(),
        "p_values": model.pvalues.round(6).to_dict(),
        "odds_ratios": odds.round(4).to_dict(),
    }


def main() -> dict:
    df = load_dataset()
    results = {
        "ols_popularity": linear_regression_analysis(df),
        "classification": classification_pipeline(df),
        "logistic_key_features": logistic_valence_popularity(df),
    }
    with open(RESULTS_DIR / "step_05_regression.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else "...")
                           for kk, vv in v.items()}
                      for k, v in results.items()}, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
