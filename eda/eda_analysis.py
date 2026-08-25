"""
BCI Motor Imagery Classification — Exploratory Data Analysis (EDA)
====================================================================
Team role: EDA & Quality Assurance.

This script performs the EDA checks requested by the project brief
(document.pdf, Section 3) on top of the artifacts already produced by
the Data & Preprocessing stage (see ../processed_data/ and
../PROCESSING_REPORT.md). It does not repeat pipeline-verification
plots that already exist in ../figures/ (raw-vs-filtered traces, PSD
filter check, artifact pie chart, channel cross-correlation,
spectrogram) — it focuses on dataset-level statistics, quality
checks, and Left-vs-Right group comparisons with significance tests.

Outputs
-------
- eda/figures/eda_01..07_*.png   (7 figures)
- eda/bandpower_stats_summary.csv
- eda/erd_laterality_summary.csv
- printed summary to stdout (also written into EDA_REPORT.md by hand)

Run from anywhere:
    python eda/eda_analysis.py
"""
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.decomposition import PCA

# ---------------------------------------------------------------- setup
sns.set_theme(style="whitegrid", context="notebook")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CHANNELS = ["FZ", "C3", "CZ", "C4"]
CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
FS = 250.0
BANDS = {"mu (8-12 Hz)": (8, 12), "beta (13-30 Hz)": (13, 30)}


def load_artifacts():
    X_clean = np.load(os.path.join(REPO, "processed_data", "X_clean.npy"))
    X_mb = np.load(os.path.join(REPO, "processed_data", "X_clean_mubeta.npy"))
    y_clean = np.load(os.path.join(REPO, "processed_data", "y_clean.npy"))
    t_epoch = np.load(os.path.join(REPO, "processed_data", "epoch_time_axis.npy"))
    audit = pd.read_csv(os.path.join(REPO, "processed_data", "trial_audit_metadata.csv"))
    raw_sample = pd.read_csv(os.path.join(REPO, "cellula_MI_data.csv"))
    return X_clean, X_mb, y_clean, t_epoch, audit, raw_sample


def bandpower(sig_1d, fs, band):
    freqs, psd = signal.welch(sig_1d, fs=fs, nperseg=min(256, len(sig_1d)))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])


# =====================================================================
# A. Class balance — before vs after cleaning
# =====================================================================
def class_balance(audit, y_clean):
    before = audit["label"].value_counts().reindex(["Left", "Right"])
    after = pd.Series(y_clean).map({0: "Left", 1: "Right"}).value_counts().reindex(["Left", "Right"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    before.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title(f"All Collected Trials (n={before.sum()})")
    axes[0].set_ylabel("Trial count")
    for i, v in enumerate(before):
        axes[0].text(i, v + 10, str(v), ha="center")

    after.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
    axes[1].set_title(f"Clean Trials After Rejection (n={after.sum()})")
    for i, v in enumerate(after):
        axes[1].text(i, v + 5, str(v), ha="center")

    plt.suptitle("Class Balance: Left vs Right Hand Motor Imagery")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_01_class_balance.png", dpi=130)
    plt.close()
    print("Class balance BEFORE cleaning:", before.to_dict())
    print("Class balance AFTER cleaning: ", after.to_dict())
    return before, after


# =====================================================================
# B1. Sampling consistency (raw timestamp jitter, sample trial)
# =====================================================================
def sampling_consistency(raw_sample):
    dt = np.diff(raw_sample["Time"].values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dt * 1000, bins=60, color="#55A868")
    ax.axvline(1000 / FS, color="red", linestyle="--", label=f"Target dt @ {FS:.0f} Hz = {1000/FS:.2f} ms")
    ax.set_xlabel("Inter-sample interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Raw Timestamp Jitter — Sample Trial (before resampling)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_02_sampling_jitter.png", dpi=130)
    plt.close()

    cv = dt.std() / dt.mean()
    n_nan = int(raw_sample.isna().sum().sum())
    print(f"Raw sample trial: {len(raw_sample)} rows, columns={list(raw_sample.columns)}, NaNs={n_nan}")
    print(f"Raw dt: mean={dt.mean()*1000:.3f} ms, std={dt.std()*1000:.3f} ms, CV={cv:.2f} "
          f"(implied fs ~{1/dt.mean():.1f} Hz, highly non-uniform -> resampling to {FS:.0f} Hz was necessary)")
    return cv, n_nan


# =====================================================================
# B2. Rejection reason breakdown
# =====================================================================
def rejection_breakdown(audit):
    def categorize(reason):
        if reason == "CLEAN":
            return []
        cats = []
        if "extreme_ptp_amplitude" in reason:
            cats.append("PTP amplitude outlier")
        if "flatline_dead_channel" in reason:
            cats.append("Flatline / dead channel")
        if "statistical_variance_outlier" in reason:
            cats.append("Variance Z-score outlier")
        return cats or ["Other"]

    reason_counts = Counter()
    for r in audit.loc[~audit["is_valid"], "rejection_reasons"]:
        for c in categorize(r):
            reason_counts[c] += 1
    reason_series = pd.Series(reason_counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    reason_series.plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_xlabel(f"Number of rejected trials (n={int((~audit['is_valid']).sum())}, categories can overlap per trial)")
    ax.set_title("Rejection Cause Breakdown")
    for i, v in enumerate(reason_series):
        ax.text(v + 5, i, str(v), va="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_03_rejection_causes.png", dpi=130)
    plt.close()
    print("Rejection causes:", dict(reason_series))
    return reason_series


# =====================================================================
# B3. Channel variance spread within the clean set, by class
# =====================================================================
def variance_by_channel_class(audit):
    clean_audit = audit[audit["is_valid"]].copy()
    var_cols = [f"var_{c}" for c in CHANNELS]
    melt = clean_audit.melt(id_vars="label", value_vars=var_cols, var_name="channel", value_name="variance")
    melt["channel"] = melt["channel"].str.replace("var_", "")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=melt, x="channel", y="variance", hue="label", ax=ax, showfliers=False)
    ax.set_title("Post-Cleaning Signal Variance by Channel and Class")
    ax.set_ylabel("Variance (uV^2)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_04_variance_by_channel_class.png", dpi=130)
    plt.close()


# =====================================================================
# C. Grand-average time-domain waveform, C3 & C4, Left vs Right
# =====================================================================
def grand_average_waveform(X_clean, y_clean, t_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, ch in zip(axes, ["C3", "C4"]):
        for label_val, label_name, color in [(0, "Left", "#4C72B0"), (1, "Right", "#DD8452")]:
            trials = X_clean[y_clean == label_val, CH_IDX[ch], :]
            mean = trials.mean(axis=0)
            std = trials.std(axis=0)
            ax.plot(t_epoch, mean, label=label_name, color=color)
            ax.fill_between(t_epoch, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(f"Channel {ch}: Grand Average (n={len(y_clean)} trials)")
        ax.set_xlabel("Time (s, post-cue)")
    axes[0].set_ylabel("Amplitude (z-scored)")
    axes[0].legend()
    plt.suptitle("Grand-Average Time-Domain Response (Broadband, Clean Trials)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_05_grand_average_waveform.png", dpi=130)
    plt.close()


# =====================================================================
# D. Band-power analysis (Welch PSD): mu/beta, Left vs Right, stats test
# =====================================================================
def bandpower_analysis(X_mb, y_clean):
    records = []
    for trial_i in range(X_mb.shape[0]):
        for ch in ["C3", "C4"]:
            sig_1d = X_mb[trial_i, CH_IDX[ch], :]
            for band_name, band in BANDS.items():
                bp = bandpower(sig_1d, FS, band)
                records.append({"trial": trial_i, "channel": ch, "band": band_name,
                                 "bandpower": bp, "label": "Left" if y_clean[trial_i] == 0 else "Right"})
    bp_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ch in zip(axes, ["C3", "C4"]):
        sub = bp_df[bp_df.channel == ch]
        sns.boxplot(data=sub, x="band", y="bandpower", hue="label", ax=ax, showfliers=False)
        ax.set_title(f"Channel {ch}: Mu/Beta Band Power by Class")
        ax.set_ylabel("Band power (a.u.)")
        ax.set_xlabel("")
    plt.suptitle("Band-Power Comparison — Left vs Right Motor Imagery (same-channel, across-class)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_06_bandpower_boxplots.png", dpi=130)
    plt.close()

    rows = []
    print("\nMann-Whitney U test, Left vs Right band power (same channel, across class):")
    for ch in ["C3", "C4"]:
        for band_name in BANDS:
            l = bp_df[(bp_df.channel == ch) & (bp_df.band == band_name) & (bp_df.label == "Left")]["bandpower"]
            r = bp_df[(bp_df.channel == ch) & (bp_df.band == band_name) & (bp_df.label == "Right")]["bandpower"]
            u, p = stats.mannwhitneyu(l, r, alternative="two-sided")
            rows.append({"channel": ch, "band": band_name, "p_value": p,
                         "left_mean": l.mean(), "right_mean": r.mean()})
            print(f"  {ch:>3} | {band_name:<15} | p={p:.4f} | Left mean={l.mean():.4f} | Right mean={r.mean():.4f}")
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(HERE, "bandpower_stats_summary.csv"), index=False)
    return bp_df, stats_df


# =====================================================================
# D2. Contralateral vs ipsilateral ERD test (neurophysiologically correct framing)
# =====================================================================
def contralateral_erd_test(X_mb, y_clean):
    records = []
    for i in range(X_mb.shape[0]):
        label = y_clean[i]  # 0=Left, 1=Right
        contra_ch = "C3" if label == 1 else "C4"  # right hand -> contralateral = left hemisphere = C3
        ipsi_ch = "C4" if label == 1 else "C3"
        for band_name, band in BANDS.items():
            contra_bp = bandpower(X_mb[i, CH_IDX[contra_ch], :], FS, band)
            ipsi_bp = bandpower(X_mb[i, CH_IDX[ipsi_ch], :], FS, band)
            records.append({"trial": i, "band": band_name, "contra": contra_bp, "ipsi": ipsi_bp})
    df = pd.DataFrame(records)

    print("\nWilcoxon signed-rank test, contralateral vs ipsilateral band power (expected: contra < ipsi = ERD):")
    rows = []
    for band_name in BANDS:
        sub = df[df.band == band_name]
        w, p = stats.wilcoxon(sub["contra"], sub["ipsi"])
        pct = 100 * (sub["ipsi"].mean() - sub["contra"].mean()) / sub["ipsi"].mean()
        rows.append({"band": band_name, "p_value": p, "contra_mean": sub["contra"].mean(),
                     "ipsi_mean": sub["ipsi"].mean(), "pct_contra_lower": pct})
        print(f"  {band_name:<15} | p={p:.4f} | contra mean={sub['contra'].mean():.4f} | "
              f"ipsi mean={sub['ipsi'].mean():.4f} | contra lower by {pct:.2f}%")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(HERE, "erd_laterality_summary.csv"), index=False)
    return out_df


# =====================================================================
# E. PCA on band-power features — class separability check
# =====================================================================
def pca_separability(bp_df):
    feat_df = bp_df.pivot_table(index=["trial", "label"], columns=["channel", "band"], values="bandpower").reset_index()
    feat_df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in feat_df.columns]
    feature_cols = [c for c in feat_df.columns if c not in ("trial", "label")]
    X_feat = np.log1p(feat_df[feature_cols].values)
    X_std = (X_feat - X_feat.mean(0)) / X_feat.std(0)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_std)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_name, color in [("Left", "#4C72B0"), ("Right", "#DD8452")]:
        mask = feat_df["label"] == label_name
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label_name, alpha=0.5, s=18, color=color)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of Mu/Beta Band-Power Features (C3, C4)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_07_pca_bandpower.png", dpi=130)
    plt.close()
    print("\nPCA explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)


def main():
    X_clean, X_mb, y_clean, t_epoch, audit, raw_sample = load_artifacts()
    print(f"X_clean: {X_clean.shape} | X_mb: {X_mb.shape} | y_clean: {y_clean.shape} | audit rows: {audit.shape[0]}\n")

    class_balance(audit, y_clean)
    print()
    sampling_consistency(raw_sample)
    print()
    rejection_breakdown(audit)
    variance_by_channel_class(audit)
    grand_average_waveform(X_clean, y_clean, t_epoch)
    bp_df, _ = bandpower_analysis(X_mb, y_clean)
    contralateral_erd_test(X_mb, y_clean)
    pca_separability(bp_df)
    print("\nAll figures saved to eda/figures/. Done.")


if __name__ == "__main__":
    main()
