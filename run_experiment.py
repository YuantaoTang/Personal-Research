"""
Evaluate how robust an LLM sentiment classifier is to token-level deletion
of the IMDB test set reviews.

Six deletion policies are compared, each parameterized by a deletion
fraction `n` (the same NOISE_LEVELS sweep used for all of them, so that
each policy removes roughly the same number of tokens from a given review
at a given level). Policies 1-5 are the baselines from the paper; policy 6
is the work-in-progress candidate for the optimal policy (RQ3):

    1. Random Deletion             - remove a random selection of tokens
    2. Frontal Deletion            - drop the front n fraction of tokens
    3. Rear Deletion                - drop the back n fraction of tokens
    4. Sentence-based Frontal Deletion - split into sentences (by punctuation),
                                          drop the front n fraction of EACH
                                          sentence's tokens
    5. Sentence-based Rear Deletion    - same, but drop the back n fraction
                                          of each sentence
    6. Length-Adaptive Deletion       - rear deletion for reviews of at most
                                          LENGTH_ADAPTIVE_THRESHOLD tokens,
                                          frontal deletion for longer ones

All deletion happens on the tokenized form (GPT token ids via tiktoken);
text is detokenized back to a normal string before being sent to the
Chat Completions API.

Outputs, all written relative to the working directory:
    results/results.png     - accuracy vs. deletion level, all policies
    results/results.txt     - the same numbers as a text table
    data/run_records.csv    - every individual sample classified: level,
                               policy, true/predicted label, original and
                               degraded token length
    data/dataset_summary.txt - positive/negative counts and review length
                                statistics for the (undegraded) test set
    results/plots/length_vs_accuracy_level_X.X.png - one plot per noise level,
        showing accuracy vs. original review length, one line per policy
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
MODEL_NAME = "gpt-4.1-mini"
SAMPLE_SIZE = 500
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROGRESS_LOG_EVERY = 50  # log a progress line every N samples within a policy
LENGTH_BIN_COUNT = 5     # number of review-length buckets for the length-vs-accuracy plots

DATA_DIR = "data"              # raw per-sample records + dataset summary
RESULTS_DIR = "results"        # aggregate accuracy table + plot
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")  # one length-vs-accuracy plot per noise level

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Silence the OpenAI client's own HTTP logging (httpx/httpcore/openai all log
# one "HTTP Request: POST ..." line per call at INFO level by default, which
# drowns out our own progress logs).
for _noisy_logger in ("httpx", "httpcore", "openai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #

def _load_encoding() -> "tiktoken.Encoding":
    """Load the tokenizer matching MODEL_NAME, with sane fallbacks."""
    try:
        return tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        pass
    for name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except Exception:  # noqa: BLE001 - fall through to next candidate
            continue
    raise RuntimeError(
        "Could not load a tiktoken encoding. This requires network access "
        "to download the tokenizer file on first use."
    )


_ENCODING = _load_encoding()


def tokenize(text: str) -> list[int]:
    """Text -> list of GPT token ids."""
    return _ENCODING.encode(text)


def detokenize(tokens: list[int]) -> str:
    """List of GPT token ids -> text. Reverses `tokenize`."""
    return _ENCODING.decode(tokens)


def compute_token_lengths(texts: pd.Series) -> pd.Series:
    """Token count for each text in `texts`, indexed the same as `texts`."""
    try:
        token_lists = _ENCODING.encode_ordinary_batch(list(texts), num_threads=8)
        lengths = [len(t) for t in token_lists]
    except AttributeError:
        # Older tiktoken versions without batch encoding: fall back to a loop.
        lengths = [len(tokenize(t)) for t in texts]
    return pd.Series(lengths, index=texts.index, name="orig_length")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', or '?' followed by whitespace."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s]


# --------------------------------------------------------------------------- #
# Deletion policies (operate on tokens; return plain detokenized text)
# --------------------------------------------------------------------------- #

def random_deletion(text: str, n: float) -> str:
    """
    Random Deletion: remove a random selection of exactly round(n * len(tokens))
    tokens (sampled without replacement), rather than flipping an independent
    coin per token. This makes the removed fraction deterministic instead of
    binomially distributed, matching the other four policies' exactness.
    """
    tokens = tokenize(text)
    num_tokens = len(tokens)
    if num_tokens == 0:
        return text

    num_to_remove = min(num_tokens - 1, int(round(num_tokens * n)))
    if num_to_remove <= 0:
        return text

    remove_indices = set(np.random.choice(num_tokens, size=num_to_remove, replace=False))
    kept = [t for i, t in enumerate(tokens) if i not in remove_indices]
    if not kept:
        kept = [tokens[np.random.randint(num_tokens)]]

    return detokenize(kept)


def frontal_deletion(text: str, n: float) -> str:
    """Frontal Deletion: drop the front n fraction of tokens; keep the tail."""
    tokens = tokenize(text)
    if not tokens:
        return text

    drop_len = min(len(tokens) - 1, int(len(tokens) * n))
    return detokenize(tokens[drop_len:])


def rear_deletion(text: str, n: float) -> str:
    """Rear Deletion: drop the back n fraction of tokens; keep the front."""
    tokens = tokenize(text)
    if not tokens:
        return text

    keep_len = max(1, int(len(tokens) * (1 - n)))
    return detokenize(tokens[:keep_len])


def sentence_frontal_deletion(text: str, n: float) -> str:
    """
    Sentence-based Frontal Deletion: split into sentences on punctuation,
    then drop the front n fraction of EACH sentence's tokens. Deletion is
    applied per-sentence (rather than to the whole review at once) so that
    roughly n fraction of every sentence is removed, not just the opening
    sentence(s).
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    kept_sentences = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        if not tokens:
            continue
        drop_len = min(len(tokens) - 1, int(len(tokens) * n)) if len(tokens) > 1 else 0
        kept = tokens[drop_len:]
        if kept:
            kept_sentences.append(detokenize(kept))

    return " ".join(kept_sentences) if kept_sentences else text


def sentence_rear_deletion(text: str, n: float) -> str:
    """Sentence-based Rear Deletion: like sentence_frontal_deletion, but
    drops the back n fraction of each sentence's tokens instead of the front."""
    sentences = split_sentences(text)
    if not sentences:
        return text

    kept_sentences = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        if not tokens:
            continue
        keep_len = max(1, int(len(tokens) * (1 - n)))
        kept = tokens[:keep_len]
        if kept:
            kept_sentences.append(detokenize(kept))

    return " ".join(kept_sentences) if kept_sentences else text


LENGTH_ADAPTIVE_THRESHOLD = 180  # tokens; see analysis in run_records.csv exploration


def length_adaptive_deletion(text: str, n: float, threshold: int = LENGTH_ADAPTIVE_THRESHOLD) -> str:
    """
    Length-Adaptive Deletion: switch between rear_deletion and frontal_deletion
    based on the review's own (undegraded) length.

    Rationale (from analysis of an earlier run's data): short reviews tend to
    state their verdict early, so trimming the END is safer; long reviews tend
    to bury the verdict after a lot of setup/synopsis, so trimming the FRONT
    is safer. Reviews at or below `threshold` tokens use rear_deletion;
    longer reviews use frontal_deletion.
    """
    tokens = tokenize(text)
    if len(tokens) <= threshold:
        return rear_deletion(text, n)
    return frontal_deletion(text, n)


DELETION_METHODS: dict[str, Callable[[str, float], str]] = {
    "random": random_deletion,
    "frontal": frontal_deletion,
    "rear": rear_deletion,
    "sentence_frontal": sentence_frontal_deletion,
    "sentence_rear": sentence_rear_deletion,
    "length_adaptive": length_adaptive_deletion,
}

POLICY_LABELS: dict[str, str] = {
    "random": "Random Deletion",
    "frontal": "Frontal Deletion",
    "rear": "Rear Deletion",
    "sentence_frontal": "Sentence-based Frontal Deletion",
    "sentence_rear": "Sentence-based Rear Deletion",
    "length_adaptive": "Length-Adaptive Deletion",
}


def apply_deletion(df: pd.DataFrame, n: float, method: str) -> pd.DataFrame:
    """Apply a deletion policy (by key in DELETION_METHODS) to every row of `df`."""
    if method not in DELETION_METHODS:
        raise ValueError(f"Unknown deletion method: {method!r}")

    df = df.copy()
    fn = DELETION_METHODS[method]
    df["text"] = df["text"].apply(lambda x: fn(x, n))
    return df


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
        client: OpenAI,
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
        )

        if self.on_call:
            self.on_call()

        result = response.choices[0].message.content.strip()
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


def evaluate(
    df: pd.DataFrame,
    classifier: SentimentClassifier,
    sample_size: int = 100,
    tracker: ProgressTracker | None = None,
    context: str = "",
    log_every: int = PROGRESS_LOG_EVERY,
    orig_lengths: pd.Series | None = None,
    indices: list | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Classify a sample of rows from `df` and return (accuracy, records).

    By default, samples `sample_size` rows via `df.sample(random_state=RANDOM_SEED)`.
    If `indices` is given instead, those exact rows are used (via `df.loc[indices]`)
    rather than drawing a new sample -- this is how a re-run can reuse the exact
    same reviews an earlier run already classified.

    `records` is a DataFrame with one row per sampled review: its original
    index, true/predicted label, whether the prediction was correct, and
    token lengths (original, pre-degradation, and degraded/as-classified).
    `orig_lengths`, if given, should be a Series mapping df's index to the
    token length of the UNDEGRADED review (see compute_token_lengths).
    """
    if indices is not None:
        sample = df.loc[indices]
    else:
        sample = df.sample(n=sample_size, random_state=RANDOM_SEED)
    actual_size = len(sample)

    preds, truths = [], []
    records = []
    for i, (idx, row) in enumerate(sample.iterrows(), start=1):
        pred_label = classifier.label_to_int(classifier.classify(row["text"]))

        correct = np.nan
        if pred_label != -1:
            preds.append(pred_label)
            truths.append(row["label"])
            correct = int(pred_label == row["label"])

        records.append(
            {
                "index": idx,
                "true_label": row["label"],
                "pred_label": pred_label,
                "correct": correct,
                "orig_length": orig_lengths.loc[idx] if orig_lengths is not None else np.nan,
                "degraded_length": len(tokenize(row["text"])),
            }
        )

        if tracker and (i % log_every == 0 or i == actual_size):
            tracker.log_progress(context=f"{context} | {i}/{actual_size} samples")

    accuracy = accuracy_score(truths, preds)
    return accuracy, pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

@dataclass
class ExperimentResults:
    noise_levels: list[float]
    random_deletion: list[float] = field(default_factory=list)
    frontal_deletion: list[float] = field(default_factory=list)
    rear_deletion: list[float] = field(default_factory=list)
    sentence_frontal_deletion: list[float] = field(default_factory=list)
    sentence_rear_deletion: list[float] = field(default_factory=list)
    length_adaptive_deletion: list[float] = field(default_factory=list)

    def as_rows(self) -> list[tuple[float, float, float, float, float, float, float]]:
        """Zip everything together as (level, random, frontal, rear,
        sentence_frontal, sentence_rear, length_adaptive) rows for reporting."""
        return list(
            zip(
                self.noise_levels,
                self.random_deletion,
                self.frontal_deletion,
                self.rear_deletion,
                self.sentence_frontal_deletion,
                self.sentence_rear_deletion,
                self.length_adaptive_deletion,
            )
        )


# Maps a DELETION_METHODS key to the ExperimentResults field name it fills.
_POLICY_TO_RESULTS_FIELD: dict[str, str] = {
    "random": "random_deletion",
    "frontal": "frontal_deletion",
    "rear": "rear_deletion",
    "sentence_frontal": "sentence_frontal_deletion",
    "sentence_rear": "sentence_rear_deletion",
    "length_adaptive": "length_adaptive_deletion",
}


def run_experiment(
    test_df: pd.DataFrame,
    classifier: SentimentClassifier,
    orig_lengths: pd.Series,
    policies: list[str] | None = None,
    indices_by_level: dict[float, list] | None = None,
) -> pd.DataFrame:
    """
    For each level n in NOISE_LEVELS, evaluate the given deletion policies
    (default: all of DELETION_METHODS) at that deletion fraction.

    `policies` lets you run only a subset (e.g. just a newly added policy),
    and `indices_by_level` lets you reuse the exact review sample an earlier
    run already used for each level instead of drawing a fresh one -- pass
    both together to extend a previous run's results.csv without repeating
    any already-computed policy's API calls.

    Returns a DataFrame with one row per individual classification made
    (used for the length-vs-accuracy plots and the raw-data export, and to
    build the aggregate ExperimentResults via build_experiment_results()).
    """
    policies = policies or list(DELETION_METHODS.keys())
    all_records: list[pd.DataFrame] = []

    total_calls_estimate = len(NOISE_LEVELS) * len(policies) * SAMPLE_SIZE
    tracker = ProgressTracker(total_calls_estimate)
    classifier.on_call = tracker.record_call  # wire real (non-cached) API calls into the tracker
    logger.info(
        "Starting experiment: %d levels x %d policies x %d samples "
        "(up to %d API calls, fewer if caching kicks in)",
        len(NOISE_LEVELS),
        len(policies),
        SAMPLE_SIZE,
        total_calls_estimate,
    )

    for level_idx, n in enumerate(NOISE_LEVELS, start=1):
        logger.info("=== Level %.1f (%d/%d) ===", n, level_idx, len(NOISE_LEVELS))
        level_indices = indices_by_level.get(round(n, 1)) if indices_by_level else None

        for policy_idx, method_key in enumerate(policies, start=1):
            label = POLICY_LABELS[method_key]
            logger.info(
                "--- %s (policy %d/%d) at level %.1f ---",
                label,
                policy_idx,
                len(policies),
                n,
            )

            degraded_df = apply_deletion(test_df, n, method_key)
            acc, records_df = evaluate(
                degraded_df,
                classifier,
                SAMPLE_SIZE,
                tracker=tracker,
                context=f"level={n:.1f} | {label}",
                orig_lengths=orig_lengths,
                indices=level_indices,
            )
            records_df["level"] = n
            records_df["policy"] = method_key
            records_df["policy_label"] = label
            all_records.append(records_df)

            logger.info("%s @ level %.1f -> accuracy=%.4f", label, n, acc)

    logger.info("Experiment complete in %s", format_duration(time.monotonic() - tracker.start_time))
    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()


def build_experiment_results(master_records: pd.DataFrame) -> ExperimentResults:
    """Derive the aggregate ExperimentResults (accuracy per level x policy)
    from a master_records DataFrame, regardless of which run(s) produced it."""
    pivot = master_records.groupby(["level", "policy"])["correct"].mean().unstack("policy")
    pivot = pivot.reindex(NOISE_LEVELS)

    results = ExperimentResults(noise_levels=NOISE_LEVELS)
    for method_key, field_name in _POLICY_TO_RESULTS_FIELD.items():
        values = pivot[method_key].tolist() if method_key in pivot.columns else [float("nan")] * len(NOISE_LEVELS)
        getattr(results, field_name).extend(values)

    return results


def plot_results(results: ExperimentResults, output_path: str = os.path.join(RESULTS_DIR, "results.png")) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(results.noise_levels, results.random_deletion, label="Random Deletion", marker="o")
    plt.plot(results.noise_levels, results.frontal_deletion, label="Frontal Deletion", marker="o")
    plt.plot(results.noise_levels, results.rear_deletion, label="Rear Deletion", marker="o")
    plt.plot(
        results.noise_levels,
        results.sentence_frontal_deletion,
        label="Sentence-based Frontal Deletion",
        marker="o",
    )
    plt.plot(
        results.noise_levels,
        results.sentence_rear_deletion,
        label="Sentence-based Rear Deletion",
        marker="o",
    )
    plt.plot(
        results.noise_levels,
        results.length_adaptive_deletion,
        label="Length-Adaptive Deletion",
        marker="o",
    )

    plt.xlabel("Deletion Level")
    plt.ylabel("Accuracy")
    plt.title("Effect of Token-Level Deletion Policies on LLM Performance")
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def write_results_txt(results: ExperimentResults, output_path: str = os.path.join(RESULTS_DIR, "results.txt")) -> None:
    """Write the accuracy results out as a human-readable table."""
    headers = [
        "Level",
        "Random Del.",
        "Frontal Del.",
        "Rear Del.",
        "Sent. Frontal Del.",
        "Sent. Rear Del.",
        "Length-Adapt. Del.",
    ]
    col_width = 20

    lines = ["Sentiment Robustness Evaluation Results", "=" * 41, ""]
    lines.append("".join(h.ljust(col_width) for h in headers))
    lines.append("-" * (col_width * len(headers)))

    for level, random_d, frontal_d, rear_d, sent_front_d, sent_rear_d, adaptive_d in results.as_rows():
        row = [
            f"{level:.1f}",
            f"{random_d:.4f}",
            f"{frontal_d:.4f}",
            f"{rear_d:.4f}",
            f"{sent_front_d:.4f}",
            f"{sent_rear_d:.4f}",
            f"{adaptive_d:.4f}",
        ]
        lines.append("".join(v.ljust(col_width) for v in row))

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


def save_run_data(master_records: pd.DataFrame, data_dir: str = DATA_DIR) -> None:
    """Save every individual classification made during the run as a CSV."""
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "run_records.csv")
    master_records.to_csv(output_path, index=False)
    logger.info("Saved %d per-sample records to %s", len(master_records), output_path)


def compute_length_bin_edges(lengths: pd.Series, n_bins: int = LENGTH_BIN_COUNT) -> np.ndarray:
    """Quantile-based bin edges over `lengths`, shared across all levels/policies
    so the length-vs-accuracy plots are directly comparable to one another."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(lengths.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        edges = np.array([lengths.min(), lengths.max() + 1])
    return edges


def plot_length_vs_accuracy(
    master_records: pd.DataFrame,
    level: float,
    bin_edges: np.ndarray,
    output_dir: str = PLOTS_DIR,
) -> None:
    """
    For one noise level, plot original review length (bucketed) against
    accuracy, with one line per deletion policy.
    """
    os.makedirs(output_dir, exist_ok=True)

    level_df = master_records[master_records["level"] == level].dropna(subset=["correct"]).copy()
    level_df["length_bin"] = pd.cut(level_df["orig_length"], bins=bin_edges, include_lowest=True)

    plt.figure()
    for method_key, label in POLICY_LABELS.items():
        policy_df = level_df[level_df["policy"] == method_key]
        if policy_df.empty:
            continue
        grouped = policy_df.groupby("length_bin", observed=True)["correct"].mean()
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

def main() -> None:
    logger.info("Using Python executable: %s", sys.executable)
    np.random.seed(RANDOM_SEED)

    load_dotenv()  # pick up OPENAI_API_KEY from a local .env file, if present
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None

    logger.info("Loading dataset")
    dataset = load_dataset("stanfordnlp/imdb")
    test_df = pd.DataFrame(dataset["test"])

    logger.info("Computing original review token lengths")
    orig_lengths = compute_token_lengths(test_df["text"])
    write_dataset_summary(test_df, orig_lengths)

    # If a previous run already saved data/run_records.csv, reuse its exact
    # per-level review sample and skip any policy that's already in it --
    # so adding a new policy only costs API calls for that new policy.
    run_records_path = os.path.join(DATA_DIR, "run_records.csv")
    existing_records: pd.DataFrame | None = None
    existing_policies: set[str] = set()
    indices_by_level: dict[float, list] | None = None

    if os.path.exists(run_records_path):
        logger.info("Found existing %s -- reusing its samples where possible", run_records_path)
        existing_records = pd.read_csv(run_records_path)
        existing_policies = set(existing_records["policy"].unique())
        indices_by_level = {
            round(float(level), 1): group["index"].unique().tolist()
            for level, group in existing_records.groupby("level")
        }
        logger.info("Policies already present: %s", sorted(existing_policies))

    remaining_policies = [k for k in DELETION_METHODS if k not in existing_policies]

    if remaining_policies:
        logger.info("Policies to run this session: %s", remaining_policies)
        if client is None:
            raise RuntimeError("Set OPENAI_API_KEY in .env (see .env.example) before running.")
        classifier = SentimentClassifier(client)
        new_records = run_experiment(
            test_df,
            classifier,
            orig_lengths,
            policies=remaining_policies,
            indices_by_level=indices_by_level,
        )
        master_records = (
            pd.concat([existing_records, new_records], ignore_index=True)
            if existing_records is not None
            else new_records
        )
    else:
        logger.info("All policies already present in %s -- skipping API calls entirely", run_records_path)
        master_records = existing_records

    results = build_experiment_results(master_records)

    plot_results(results)
    write_results_txt(results)
    save_run_data(master_records)

    length_bin_edges = compute_length_bin_edges(orig_lengths)
    for n in NOISE_LEVELS:
        plot_length_vs_accuracy(master_records, n, length_bin_edges)


if __name__ == "__main__":
    main()