"""
Learn an optimal deletion policy as a continuous priority function over tokens.

Model
-----
Each record is one degraded review r (N_r original tokens, keep-mask m_r) and
whether the LLM classified it correctly. Suppose each token i carries an
importance w(z_i) that depends only on where it sits, z_i = (position in the
review, position in its sentence, review length); see policies.PriorityModel.
The importance the degraded review retains is

    R(r) = (1 / N_r) * sum over kept tokens i of w(z_i)

and the model is

    P(correct | r) = sigmoid(a + b * length_z(r) + R(r)).

w is a weighted sum of fixed smooth basis functions phi_k, so R is linear in
the weights: R(r) = beta . F_r with F_r = (1 / N_r) * sum over kept i of phi(z_i).
Fitting w is therefore an (L2-regularized) logistic regression of `correct` on
the per-record features F_r, computed from the keep-masks run_experiment.py
stores.

Policy
------
Under this model, deleting a set D of tokens lowers R by (1/N) * sum over D of
w(z_i), and sigmoid is increasing, so for any deletion budget the deletion that
maximizes predicted accuracy is: delete the tokens with the lowest w first.
The fitted w is therefore used directly as the learned policy's priority
function.

Model selection uses cross-validation grouped by review (all records of one
review fall in the same fold), so the score measures how well the fitted
function generalizes to reviews it has not seen.

Usage
-----
    python optimize_policy.py fit        # fit on data/train_records.csv
    python optimize_policy.py evaluate   # compare policies on data/holdout_records.csv

Outputs
-------
    results/learned_policy/policy.json            - fitted priority function
                                                    (read by run_experiment.py)
    results/learned_policy/priority_function.png  - w against position
    results/learned_policy/fit_report.txt         - model comparison, chosen
        settings, and predicted vs. observed accuracy per policy (train reviews)
    results/holdout/comparison.txt                - written by `evaluate`
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from policies import BASELINE_POLICIES, PriorityFn, PriorityModel, ReviewTokens, apply_policy, decode_mask
from run_experiment import (
    EXPERIMENT_VERSION,
    LEARNED_POLICY_LABEL,
    LEARNED_POLICY_PATH,
    NOISE_LEVELS,
    RESULTS_DIR,
    accuracy_table,
    learned_policy_key,
    load_split,
    read_records,
    records_path,
)

# (position degree, sentence-position degree) candidates for the Bernstein
# polynomials; the best is chosen by grouped cross-validation.
DEGREE_GRID = [(2, 2), (4, 2), (6, 3), (8, 3)]
C_GRID = np.logspace(-2, 3, 11)  # inverse L2 regularization strengths to try
CV_FOLDS = 5
SIGNIFICANCE = 0.05

OUTPUT_DIR = os.path.dirname(LEARNED_POLICY_PATH)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_records(split: str) -> pd.DataFrame:
    path = records_path(split)
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found. Run `python run_experiment.py --split {split}` first.")
    records = read_records(path)
    if set(records.get("version", pd.Series([1])).fillna(1).astype(int)) != {EXPERIMENT_VERSION}:
        raise RuntimeError(f"{path} was not produced by experiment version {EXPERIMENT_VERSION}.")
    return records.dropna(subset=["correct"]).reset_index(drop=True)


def length_stats(reviews: dict[int, ReviewTokens]) -> tuple[float, float]:
    log_lengths = np.log([max(r.num_tokens, 1) for r in reviews.values()])
    return float(log_lengths.mean()), float(log_lengths.std())


def mask_features(spec: PriorityModel, review: ReviewTokens, keep: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Regression features of one degraded review: the kept tokens' basis
    values summed and divided by the ORIGINAL length, plus length_z."""
    basis = spec.token_basis(review) if basis is None else basis
    return np.r_[basis[keep].sum(axis=0) / max(review.num_tokens, 1), spec.length_z(review.num_tokens)]


def record_features(records: pd.DataFrame, reviews: dict[int, ReviewTokens], spec: PriorityModel) -> np.ndarray:
    """Feature matrix for every record, built from its stored keep-mask."""
    bases = {idx: spec.token_basis(review) for idx, review in reviews.items()}
    rows = []
    for idx, hex_mask in zip(records["index"], records["kept_mask"]):
        review = reviews[int(idx)]
        rows.append(mask_features(spec, review, decode_mask(hex_mask, review.num_tokens), bases[int(idx)]))
    return np.array(rows)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

class FittedRegression:
    """Standardize -> L2 logistic regression, with C chosen by grouped CV."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        folds = list(GroupKFold(n_splits=CV_FOLDS).split(X, y, groups))
        self.clf = LogisticRegressionCV(Cs=C_GRID, cv=folds, scoring="neg_log_loss", max_iter=10000)
        self.clf.fit(self.scaler.transform(X), y)
        self.cv_log_loss = float(-self.clf.scores_[1].mean(axis=0).max())

    @property
    def raw_coefficients(self) -> np.ndarray:
        """Coefficients on the unstandardized features (centering only moves
        the intercept, which does not affect the priority order)."""
        return self.clf.coef_[0] / self.scaler.scale_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def null_log_loss(y: np.ndarray, groups: np.ndarray) -> float:
    """Grouped-CV log loss of predicting each fold's training-set accuracy."""
    losses = []
    for train, test in GroupKFold(n_splits=CV_FOLDS).split(y, y, groups):
        p = np.clip(y[train].mean(), 1e-6, 1 - 1e-6)
        losses.append(-np.mean(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p)))
    return float(np.mean(losses))


def predicted_accuracy(
    spec: PriorityModel, regression: FittedRegression, reviews: dict[int, ReviewTokens], priority: PriorityFn, n: float
) -> float:
    """Model-predicted accuracy of a policy at level n over the given reviews."""
    X = np.array([mask_features(spec, r, apply_policy(r, n, priority)[1]) for r in reviews.values()])
    return float(regression.predict(X).mean())


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #

def fit() -> None:
    records = load_records("train")
    _, reviews = load_split("train")
    y = records["correct"].astype(int).to_numpy()
    groups = records["index"].to_numpy()
    log_mean, log_std = length_stats(reviews)
    logger.info("Fitting on %d records from %d reviews", len(records), records["index"].nunique())

    # Reference models: base rate only, and "how much was kept" + length
    # (ignores WHERE tokens were deleted, so it cannot tell policies apart).
    kept_fraction = np.array(
        [decode_mask(h, reviews[int(i)].num_tokens).mean() for i, h in zip(records["index"], records["kept_mask"])]
    )
    length_z = np.array([(np.log(reviews[int(i)].num_tokens) - log_mean) / log_std for i in records["index"]])
    comparisons = {
        "base rate only": null_log_loss(y, groups),
        "kept fraction + length (position-blind)": FittedRegression(
            np.c_[kept_fraction, length_z], y, groups
        ).cv_log_loss,
    }

    best: tuple[float, PriorityModel, FittedRegression] | None = None
    for degree_position, degree_sentence in DEGREE_GRID:
        spec = PriorityModel(degree_position, degree_sentence, log_mean, log_std)
        regression = FittedRegression(record_features(records, reviews, spec), y, groups)
        comparisons[f"priority model, degrees {degree_position}/{degree_sentence}"] = regression.cv_log_loss
        logger.info("degrees %d/%d: CV log loss %.5f", degree_position, degree_sentence, regression.cv_log_loss)
        if best is None or regression.cv_log_loss < best[0]:
            best = (regression.cv_log_loss, spec, regression)

    _, spec, regression = best
    model = spec.with_coefficients(regression.raw_coefficients[:-1])  # last coefficient is length_z
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(
        LEARNED_POLICY_PATH,
        metadata={
            "experiment_version": EXPERIMENT_VERSION,
            "train_records": len(records),
            "train_reviews": int(records["index"].nunique()),
            "cv_log_loss": best[0],
            "C": float(regression.clf.C_[0]),
            "intercept_standardized": float(regression.clf.intercept_[0]),
            "length_coefficient": float(regression.raw_coefficients[-1]),
        },
    )
    logger.info("Saved learned policy %s to %s", learned_policy_key(model), LEARNED_POLICY_PATH)

    plot_priority_function(model, reviews)
    write_fit_report(model, spec, regression, comparisons, records, reviews)


def plot_priority_function(model: PriorityModel, reviews: dict[int, ReviewTokens]) -> None:
    """Plot the fitted importance against position. Only the ORDER of
    importance matters: tokens lower on these curves are deleted first."""
    x = np.linspace(0, 1, 201)
    lengths = np.percentile([r.num_tokens for r in reviews.values()], [10, 50, 90]).astype(int)
    k = model.degree_position + 1

    fig, (ax_pos, ax_sent) = plt.subplots(1, 2, figsize=(11, 4))
    for length, name in zip(lengths, ("short", "median", "long")):
        basis = model.basis(x, np.zeros_like(x), int(length))
        coef = np.r_[model.coef_position, np.zeros(model.degree_sentence + 1), model.coef_position_x_length]
        ax_pos.plot(x, basis @ coef, label=f"{name} review ({length} tokens)")
    ax_pos.set_xlabel("Position in review (0 = start, 1 = end)")
    ax_pos.set_ylabel("Importance w (higher = kept longer)")
    ax_pos.set_title("Review-position component")
    ax_pos.legend()

    sentence_basis = model.basis(np.zeros_like(x), x, int(lengths[1]))[:, k : k + model.degree_sentence + 1]
    ax_sent.plot(x, sentence_basis @ np.asarray(model.coef_sentence))
    ax_sent.set_xlabel("Position in sentence (0 = start, 1 = end)")
    ax_sent.set_title("Sentence-position component")

    fig.suptitle(f"Learned priority function ({learned_policy_key(model)})")
    path = os.path.join(OUTPUT_DIR, "priority_function.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def write_fit_report(
    model: PriorityModel,
    spec: PriorityModel,
    regression: FittedRegression,
    comparisons: dict[str, float],
    records: pd.DataFrame,
    reviews: dict[int, ReviewTokens],
) -> None:
    observed = accuracy_table(records)
    priorities: dict[str, PriorityFn] = {label: fn for label, fn in BASELINE_POLICIES.values()}
    priorities[LEARNED_POLICY_LABEL] = model

    lines = [
        "Learned Deletion Policy: Fit Report",
        "=" * 40,
        "",
        f"Policy id: {learned_policy_key(model)}",
        f"Trained on {len(records)} records from {records['index'].nunique()} reviews (train split).",
        f"Chosen degrees: position {spec.degree_position}, sentence position {spec.degree_sentence}; "
        f"L2 C = {regression.clf.C_[0]:.3g}",
        "",
        f"Cross-validated log loss ({CV_FOLDS}-fold, grouped by review; lower is better):",
    ]
    lines += [f"  {name:<45}{loss:.5f}" for name, loss in comparisons.items()]
    lines += [
        "",
        "If the priority model does not beat the position-blind model, WHERE tokens",
        "are deleted carries no detectable signal and the learned policy is noise.",
        "",
        "Predicted accuracy on the train reviews (model), next to observed accuracy",
        "where available. The learned policy's number is in-sample and optimistic;",
        "only the holdout split gives an unbiased comparison.",
        "",
    ]

    header = "Level".ljust(7) + "".join(label.replace(" Deletion", "")[:22].ljust(24) for label in priorities)
    lines += [header, "-" * len(header)]
    for n in NOISE_LEVELS:
        cells = []
        for label, priority in priorities.items():
            predicted = predicted_accuracy(spec, regression, reviews, priority, n)
            seen = observed.at[n, label] if label in observed.columns and n in observed.index else None
            cells.append(f"{predicted:.3f}" + (f" ({seen:.3f})" if seen is not None else ""))
        lines.append(f"{n:.1f}".ljust(7) + "".join(c.ljust(24) for c in cells))

    # Where does the learned policy delete first? Position deciles of the
    # tokens it removes at level 0.5, for short vs. long reviews.
    median = np.median([r.num_tokens for r in reviews.values()])
    lines += ["", "Share of tokens deleted at level 0.5 by learned policy, per review-position decile:"]
    for name, group in (("short (< median)", lambda r: r.num_tokens < median), ("long (>= median)", lambda r: r.num_tokens >= median)):
        positions = np.concatenate(
            [r.position[~apply_policy(r, 0.5, model)[1]] for r in reviews.values() if group(r)]
        )
        shares = np.histogram(positions, bins=10, range=(0, 1))[0] / max(len(positions), 1)
        lines.append(f"  {name:<18}" + " ".join(f"{s:4.0%}" for s in shares))
    lines.append("  " + " " * 18 + " ".join(f"{d / 10:.1f}+".rjust(4) for d in range(10)))

    path = os.path.join(OUTPUT_DIR, "fit_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote %s", path)
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #

def evaluate() -> None:
    """Compare the learned policy with every baseline on the holdout reviews,
    level by level, using an exact McNemar test on the paired outcomes."""
    records = load_records("holdout")
    model = PriorityModel.load(LEARNED_POLICY_PATH)
    learned = learned_policy_key(model)
    if learned not in set(records["policy"]):
        raise RuntimeError(
            f"No holdout records for the current learned policy {learned}. "
            "Run `python run_experiment.py --split holdout` first."
        )

    outcome = records.pivot_table(index=["level", "index"], columns="policy", values="correct")
    lines = [
        f"Holdout comparison: {LEARNED_POLICY_LABEL} ({learned}) vs. baselines",
        "=" * 60,
        "",
        f"Each cell: accuracy difference (learned - baseline), exact McNemar p-value.",
        f"* = significant at p < {SIGNIFICANCE}.",
        "",
    ]
    header = "Level".ljust(7) + "Learned".ljust(10) + "".join(label.replace(" Deletion", "")[:20].ljust(24) for label, _ in BASELINE_POLICIES.values())
    lines += [header, "-" * len(header)]

    wins = {key: [0, 0] for key in BASELINE_POLICIES}  # [significantly better, significantly worse]
    for n in NOISE_LEVELS:
        level = outcome.loc[outcome.index.get_level_values("level") == n]
        cells = []
        for key in BASELINE_POLICIES:
            pair = level[[learned, key]].dropna()
            better = int(((pair[learned] == 1) & (pair[key] == 0)).sum())
            worse = int(((pair[learned] == 0) & (pair[key] == 1)).sum())
            p = binomtest(better, better + worse).pvalue if better + worse else 1.0
            diff = pair[learned].mean() - pair[key].mean()
            star = "*" if p < SIGNIFICANCE else " "
            if p < SIGNIFICANCE:
                wins[key][0 if diff > 0 else 1] += 1
            cells.append(f"{diff:+.3f} p={p:.3f}{star}")
        accuracy = level[learned].mean()
        lines.append(f"{n:.1f}".ljust(7) + f"{accuracy:.3f}".ljust(10) + "".join(c.ljust(24) for c in cells))

    lines += ["", "Levels where the learned policy is significantly better / worse:"]
    lines += [f"  vs. {label:<34}{wins[key][0]} better, {wins[key][1]} worse" for key, (label, _) in BASELINE_POLICIES.items()]

    out_dir = os.path.join(RESULTS_DIR, "holdout")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comparison.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    logger.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Learn and evaluate a priority-based deletion policy.")
    parser.add_argument("command", choices=("fit", "evaluate"))
    args = parser.parse_args(argv)
    fit() if args.command == "fit" else evaluate()


if __name__ == "__main__":
    main()
