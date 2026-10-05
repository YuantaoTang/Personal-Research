"""
Evaluate how robust an LLM sentiment classifier is to token-level deletion
of IMDB test-set reviews, and collect the data optimize_policy.py uses to
learn an optimal deletion policy.

Every deletion policy is a priority function over a review's tokens (see
policies.py): at deletion level n, the round(n * N) lowest-priority tokens of
an N-token review are deleted, so all policies remove the same number of
tokens from a given review at a given level. Policies compared:

    1. Random Deletion                 - a fixed random deletion order per review
    2. Frontal Deletion                - front of the review first
    3. Rear Deletion                   - back of the review first
    4. Sentence-based Frontal Deletion - front of EACH sentence first (sentences
                                         split on punctuation and <br /> breaks)
    5. Sentence-based Rear Deletion    - back of each sentence first
    6. Length-Adaptive Deletion        - Rear for reviews of <= 180 tokens,
                                         Frontal for longer ones
    7. Learned Priority Deletion       - priority = w(token), a continuous
                                         function fitted by optimize_policy.py
                                         (holdout split only)

Two disjoint review samples ("splits") are used:
    train   - the 500 reviews of the original study; baseline policies only.
              optimize_policy.py fits the learned policy on these records.
    holdout - 500 different reviews; baselines plus the learned policy, for an
              out-of-sample comparison (hypothesis H3.1).

Each record stores the deletion mask (which of the review's original tokens
were kept), so the regression can use any token-level feature.

Usage:
    python run_experiment.py                   # 1. collect train data
    python optimize_policy.py fit              # 2. learn the priority function
    python run_experiment.py --split holdout   # 3. evaluate on unseen reviews
    python optimize_policy.py evaluate         # 4. paired significance tests

Outputs, all written relative to the working directory:
    data/{split}_records.csv  - every individual classification: level, policy,
                                true/predicted label, original and degraded
                                token length, and the keep-mask
    data/dataset_summary.txt  - positive/negative counts and review length
                                statistics for the (undegraded) test set
    results/{split}/results.txt, results.png - accuracy per level and policy
    results/{split}/plots/length_vs_accuracy_level_X.X.png - one plot per
        level: accuracy vs. original review length, one line per policy

Records are tagged with EXPERIMENT_VERSION. A rerun resumes from any
(level, policy) blocks already recorded under the current version and refuses
to mix in records from an older version.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from policies import (
    BASELINE_POLICIES,
    MODEL_NAME,
    RANDOM_SEED,
    PriorityFn,
    PriorityModel,
    ReviewTokens,
    apply_policy,
    encode_mask,
    prepare_review,
    tokenize,
    tokenize_batch,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Bump whenever a change would alter the text sent to the model or how it is
# classified, so records produced by older code are never reused.
#   v1: original run (paper draft numbers; archived in archive/v1/)
#   v2: temperature 0, <br />-aware sentence splitting, exact deletion budgets
#   v3: all policies as priority functions; keep-masks stored; holdout split
EXPERIMENT_VERSION = 3

TEMPERATURE = 0.0
MAX_API_RETRIES = 5
SAMPLE_SIZE = 500    # reviews per split
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROGRESS_LOG_EVERY = 50  # log a progress line every N samples within a block
LENGTH_BIN_COUNT = 5     # number of review-length buckets for the length-vs-accuracy plots

SPLITS = ("train", "holdout")
DATA_DIR = "data"
RESULTS_DIR = "results"
LEARNED_POLICY_PATH = os.path.join(RESULTS_DIR, "learned_policy", "policy.json")
LEARNED_POLICY_LABEL = "Learned Priority Deletion"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Silence the OpenAI client's own HTTP logging (httpx/httpcore/openai all log
# one "HTTP Request: POST ..." line per call at INFO level by default, which
# drowns out our own progress logs).
for _noisy_logger in ("httpx", "httpcore", "openai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


def records_path(split: str) -> str:
    return os.path.join(DATA_DIR, f"{split}_records.csv")


def read_records(path: str) -> pd.DataFrame:
    """Read a records CSV, keeping keep-masks as strings (an all-digit hex mask
    would otherwise be parsed as a number)."""
    return pd.read_csv(path, dtype={"kept_mask": str})


# --------------------------------------------------------------------------- #
# Review samples and policies
# --------------------------------------------------------------------------- #

def select_sample(test_df: pd.DataFrame, split: str) -> pd.DataFrame:
    """The fixed review sample for a split. `train` is the original study's
    sample; `holdout` is drawn from the remaining reviews, so the two are disjoint."""
    train = test_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    if split == "train":
        return train
    return test_df.drop(train.index).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED + 1)


def learned_policy_key(model: PriorityModel) -> str:
    """Policy key for a fitted model. Includes its fingerprint, so refitting
    the model yields a new policy instead of mixing records of two functions."""
    return f"learned_{model.fingerprint()}"


def policies_for_split(split: str) -> dict[str, tuple[str, PriorityFn]]:
    """Baselines for both splits; the learned policy only on holdout, where it
    can be evaluated on reviews it was not fitted on."""
    policies = dict(BASELINE_POLICIES)
    if split == "holdout":
        if not os.path.exists(LEARNED_POLICY_PATH):
            raise RuntimeError(
                f"{LEARNED_POLICY_PATH} not found. Run the train split and "
                "`python optimize_policy.py fit` before the holdout split."
            )
        model = PriorityModel.load(LEARNED_POLICY_PATH)
        policies[learned_policy_key(model)] = (LEARNED_POLICY_LABEL, model)
    return policies


# --------------------------------------------------------------------------- #
# Progress tracking
# --------------------------------------------------------------------------- #

def format_duration(seconds: float) -> str:
    """Render a duration in seconds as e.g. '1h02m03s'."""
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressTracker:
    """Tracks completed API calls against an expected total and logs an ETA."""

    def __init__(self, total_calls: int):
        self.total_calls = total_calls
        self.completed = 0
        self.start_time = time.monotonic()

    def record_call(self) -> None:
        """Call this once per REAL API call (not per cache hit)."""
        self.completed += 1

    def log_progress(self, context: str = "") -> None:
        elapsed = time.monotonic() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining_calls = max(0, self.total_calls - self.completed)
        eta_seconds = remaining_calls / rate if rate > 0 else float("inf")
        pct = 100 * self.completed / self.total_calls if self.total_calls else 0.0

        logger.info(
            "%s%d/%d API calls (%.1f%%) | elapsed=%s | est. remaining=%s",
            f"[{context}] " if context else "",
            self.completed,
            self.total_calls,
            pct,
            format_duration(elapsed),
            format_duration(eta_seconds),
        )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

class SentimentClassifier:
    """Thin wrapper around the OpenAI chat API with response caching."""

    def __init__(
        self,
        client: OpenAI | None,
        model: str = MODEL_NAME,
        on_call: Callable[[], None] | None = None,
    ):
        self.client = client
        self.model = model
        self._cache: dict[str, str] = {}
        self.on_call = on_call  # invoked once per real (non-cached) API call

    def classify(self, text: str) -> str:
        if text in self._cache:
            return self._cache[text]

        prompt = (
            "Classify this movie review as Positive or Negative.\n\n"
            f"Review: {text}\n\n"
            "Answer with only one word: Positive or Negative."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=TEMPERATURE,
        )

        if self.on_call:
            self.on_call()

        result = (response.choices[0].message.content or "").strip()
        self._cache[text] = result
        return result

    @staticmethod
    def label_to_int(output: str) -> int:
        output = output.strip().lower()
        if "positive" in output:
            return 1
        if "negative" in output:
            return 0
        return -1  # fallback for unparsable output


def evaluate_block(
    reviews: dict[int, ReviewTokens],
    labels: pd.Series,
    n: float,
    priority: PriorityFn,
    classifier: SentimentClassifier,
    tracker: ProgressTracker | None = None,
    context: str = "",
) -> tuple[float, pd.DataFrame]:
    """
    Degrade every review at level `n` under `priority`, classify it, and return
    (accuracy, records). Unparsable model outputs are excluded from the
    accuracy and recorded with pred_label=-1 and correct=NaN.
    """
    records = []
    for i, (idx, review) in enumerate(reviews.items(), start=1):
        text, keep = apply_policy(review, n, priority)
        pred_label = classifier.label_to_int(classifier.classify(text))
        true_label = int(labels.loc[idx])
        records.append(
            {
                "index": idx,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": int(pred_label == true_label) if pred_label != -1 else np.nan,
                "orig_length": review.num_tokens,
                "degraded_length": len(tokenize(text)),
                "kept_mask": encode_mask(keep),
            }
        )
        if tracker and (i % PROGRESS_LOG_EVERY == 0 or i == len(reviews)):
            tracker.log_progress(context=f"{context} | {i}/{len(reviews)} samples")

    records_df = pd.DataFrame(records)
    return records_df["correct"].mean(), records_df


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

def completed_blocks(records: pd.DataFrame | None) -> set[tuple[float, str]]:
    """(level, policy) pairs that already have a full sample's worth of records."""
    if records is None or records.empty:
        return set()
    counts = records.groupby(["level", "policy"]).size()
    return {(round(float(level), 1), policy) for (level, policy), c in counts.items() if c >= SAMPLE_SIZE}


def run_experiment(
    reviews: dict[int, ReviewTokens],
    labels: pd.Series,
    policies: dict[str, tuple[str, PriorityFn]],
    classifier: SentimentClassifier,
    existing_records: pd.DataFrame | None = None,
    on_block_done: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """
    For each level n in NOISE_LEVELS and each policy, degrade every review and
    classify it.

    (level, policy) blocks already present in `existing_records` are skipped,
    so a run that crashed or was interrupted resumes where it stopped, and a
    newly added policy only costs API calls for that policy. After each block,
    the combined records so far are passed to `on_block_done` (used to
    checkpoint the records CSV).

    Returns a DataFrame with one row per individual classification (existing
    plus new).
    """
    done = completed_blocks(existing_records)
    todo = [(n, key) for n in NOISE_LEVELS for key in policies if (round(n, 1), key) not in done]
    all_records: list[pd.DataFrame] = []
    if existing_records is not None and not existing_records.empty:
        all_records.append(existing_records)

    if not todo:
        logger.info("All level/policy blocks already recorded -- skipping API calls entirely")
        return pd.concat(all_records, ignore_index=True)

    total_calls_estimate = len(todo) * len(reviews)
    tracker = ProgressTracker(total_calls_estimate)
    classifier.on_call = tracker.record_call  # wire real (non-cached) API calls into the tracker
    logger.info(
        "Starting experiment: %d of %d level/policy blocks to run x %d samples "
        "(up to %d API calls, fewer if caching kicks in)",
        len(todo),
        len(NOISE_LEVELS) * len(policies),
        len(reviews),
        total_calls_estimate,
    )

    for block_idx, (n, key) in enumerate(todo, start=1):
        label, priority = policies[key]
        logger.info("--- %s at level %.1f (block %d/%d) ---", label, n, block_idx, len(todo))

        acc, records_df = evaluate_block(
            reviews, labels, n, priority, classifier, tracker=tracker, context=f"level={n:.1f} | {label}"
        )
        records_df["level"] = n
        records_df["policy"] = key
        records_df["policy_label"] = label
        records_df["version"] = EXPERIMENT_VERSION
        all_records.append(records_df)

        logger.info("%s @ level %.1f -> accuracy=%.4f", label, n, acc)
        if on_block_done:
            on_block_done(pd.concat(all_records, ignore_index=True))

    logger.info("Experiment complete in %s", format_duration(time.monotonic() - tracker.start_time))
    return pd.concat(all_records, ignore_index=True)


def accuracy_table(records: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per level (rows) and policy label (columns), in recording order."""
    table = records.groupby(["level", "policy_label"])["correct"].mean().unstack("policy_label")
    return table[list(dict.fromkeys(records["policy_label"]))]


def plot_results(table: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    for label in table.columns:
        plt.plot(table.index, table[label], label=label, marker="o")
    plt.xlabel("Deletion Level")
    plt.ylabel("Accuracy")
    plt.title("Effect of Token-Level Deletion Policies on LLM Performance")
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def write_results_txt(table: pd.DataFrame, output_path: str) -> None:
    """Write the accuracy results out as a human-readable table."""
    col_width = 34
    lines = ["Sentiment Robustness Evaluation Results", "=" * 41, ""]
    lines.append("Level".ljust(8) + "".join(label.ljust(col_width) for label in table.columns))
    lines.append("-" * (8 + col_width * len(table.columns)))
    for level, row in table.iterrows():
        lines.append(f"{level:.1f}".ljust(8) + "".join(f"{v:.4f}".ljust(col_width) for v in row))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote results to %s", output_path)


def write_dataset_summary(
    test_df: pd.DataFrame,
    orig_lengths: pd.Series,
    data_dir: str = DATA_DIR,
) -> None:
    """Write a text summary of the (undegraded) test set: class balance and
    review length statistics, measured in GPT tokens."""
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "dataset_summary.txt")

    n_total = len(test_df)
    n_positive = int((test_df["label"] == 1).sum())
    n_negative = int((test_df["label"] == 0).sum())

    lines = [
        "IMDB Test Set Summary",
        "=" * 30,
        "",
        f"Total reviews:    {n_total}",
        f"Positive reviews: {n_positive} ({n_positive / n_total:.1%})",
        f"Negative reviews: {n_negative} ({n_negative / n_total:.1%})",
        "",
        "Review length, in GPT tokens (original text, before any deletion):",
        f"  Mean:    {orig_lengths.mean():.2f}",
        f"  Median:  {orig_lengths.median():.2f}",
        f"  Std Dev: {orig_lengths.std():.2f}",
        f"  Min:     {orig_lengths.min():.0f}",
        f"  Max:     {orig_lengths.max():.0f}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote dataset summary to %s", output_path)


def compute_token_lengths(texts: pd.Series) -> pd.Series:
    """Token count for each text in `texts`, indexed the same as `texts`."""
    lengths = [len(t) for t in tokenize_batch(list(texts))]
    return pd.Series(lengths, index=texts.index, name="orig_length")


def save_records(records: pd.DataFrame, path: str) -> None:
    """Save every individual classification made during the run as a CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records.to_csv(path, index=False)
    logger.info("Saved %d per-sample records to %s", len(records), path)


def compute_length_bin_edges(lengths: pd.Series, n_bins: int = LENGTH_BIN_COUNT) -> np.ndarray:
    """Quantile-based bin edges over `lengths`, shared across all levels/policies
    so the length-vs-accuracy plots are directly comparable to one another."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(lengths.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        edges = np.array([lengths.min(), lengths.max() + 1])
    return edges


def plot_length_vs_accuracy(
    records: pd.DataFrame,
    level: float,
    bin_edges: np.ndarray,
    output_dir: str,
) -> None:
    """
    For one noise level, plot original review length (bucketed) against
    accuracy, with one line per deletion policy.
    """
    os.makedirs(output_dir, exist_ok=True)

    level_df = records[np.isclose(records["level"], level)].dropna(subset=["correct"]).copy()
    level_df["length_bin"] = pd.cut(level_df["orig_length"], bins=bin_edges, include_lowest=True)

    plt.figure()
    for label in dict.fromkeys(level_df["policy_label"]):
        grouped = level_df[level_df["policy_label"] == label].groupby("length_bin", observed=True)["correct"].mean()
        x = [interval.mid for interval in grouped.index]
        plt.plot(x, grouped.to_numpy(), marker="o", label=label)

    plt.xlabel("Original Review Length (tokens)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title(f"Review Length vs. Accuracy (Deletion Level = {level:.1f})")
    plt.legend()

    output_path = os.path.join(output_dir, f"length_vs_accuracy_level_{level:.1f}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved length-vs-accuracy plot for level %.1f to %s", level, output_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def load_split(split: str) -> tuple[pd.DataFrame, dict[int, ReviewTokens]]:
    """Load the IMDB test set and tokenize the given split's review sample.
    Returns (full test set, {review index: ReviewTokens})."""
    logger.info("Loading dataset")
    test_df = pd.DataFrame(load_dataset("stanfordnlp/imdb")["test"])
    sample_df = select_sample(test_df, split)
    reviews = {int(idx): prepare_review(idx, text) for idx, text in sample_df["text"].items()}
    return test_df, reviews


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=SPLITS, default="train", help="review sample to run on")
    args = parser.parse_args(argv)
    split = args.split

    logger.info("Using Python executable: %s", sys.executable)
    load_dotenv()  # pick up OPENAI_API_KEY from a local .env file, if present
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, max_retries=MAX_API_RETRIES) if api_key else None

    test_df, reviews = load_split(split)
    logger.info("Computing original review token lengths")
    write_dataset_summary(test_df, compute_token_lengths(test_df["text"]))

    policies = policies_for_split(split)
    path = records_path(split)

    # Resume from the records CSV if a previous run (or an interrupted one)
    # already recorded some level/policy blocks under this version.
    existing_records: pd.DataFrame | None = None
    if os.path.exists(path):
        existing_records = read_records(path)
        versions = set(existing_records.get("version", pd.Series([1])).fillna(1).astype(int))
        if versions != {EXPERIMENT_VERSION}:
            raise RuntimeError(
                f"{path} holds records from experiment version(s) {sorted(versions)}, "
                f"but this code is version {EXPERIMENT_VERSION}. Move or delete that file so "
                "results from different code versions are not mixed."
            )
        if set(existing_records["index"]) != set(reviews):
            raise RuntimeError(f"{path} was recorded on a different review sample.")
        logger.info("Found existing %s -- resuming from its recorded blocks", path)

    done = completed_blocks(existing_records)
    todo = sum((round(n, 1), key) not in done for n in NOISE_LEVELS for key in policies)
    if todo and client is None:
        raise RuntimeError("Set OPENAI_API_KEY in .env (see .env.example) before running.")

    records = run_experiment(
        reviews,
        test_df["label"],
        policies,
        SentimentClassifier(client),
        existing_records=existing_records,
        on_block_done=lambda r: save_records(r, path),  # checkpoint after every block
    )
    save_records(records, path)

    out_dir = os.path.join(RESULTS_DIR, split)
    table = accuracy_table(records)
    plot_results(table, os.path.join(out_dir, "results.png"))
    write_results_txt(table, os.path.join(out_dir, "results.txt"))

    length_bin_edges = compute_length_bin_edges(pd.Series([r.num_tokens for r in reviews.values()]))
    for n in NOISE_LEVELS:
        plot_length_vs_accuracy(records, n, length_bin_edges, os.path.join(out_dir, "plots"))


if __name__ == "__main__":
    main()
