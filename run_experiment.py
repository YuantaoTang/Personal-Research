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
    4. Sentence-based Frontal Deletion - split into sentences (by punctuation
                                          and <br /> line breaks), drop the
                                          front n fraction of EACH sentence's
                                          tokens
    5. Sentence-based Rear Deletion    - same, but drop the back n fraction
                                          of each sentence
    6. Length-Adaptive Deletion       - rear deletion for reviews of at most
                                          LENGTH_ADAPTIVE_THRESHOLD tokens,
                                          frontal deletion for longer ones

All deletion happens on the tokenized form (GPT token ids via tiktoken);
text is detokenized back to a normal string before being sent to the
Chat Completions API. Every policy removes round(n * len(tokens)) tokens
from a review, so policies are compared at equal deletion budgets.

Records in data/run_records.csv are tagged with EXPERIMENT_VERSION. A rerun
resumes from any (level, policy) blocks already recorded under the current
version and refuses to mix in records from an older version.

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

# Bump whenever a change would alter the text sent to the model or how it is
# classified, so records produced by older code are never reused.
#   v1: original run (paper draft numbers; archived in archive/v1/)
#   v2: temperature 0, <br />-aware sentence splitting, exact deletion budgets
EXPERIMENT_VERSION = 2

RANDOM_SEED = 42
MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.0
MAX_API_RETRIES = 5
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


# A sentence boundary is either whitespace after '.', '!' or '?', or a run of
# HTML line breaks. IMDB reviews mark paragraphs with "<br /><br />", often
# directly after the punctuation ("...end.<br /><br />Next..."), so splitting
# on punctuation + whitespace alone merges sentences across paragraphs.
_SENTENCE_SEPARATOR_RE = re.compile(r"((?<=[.!?])\s+|\s*(?:<br\s*/?>\s*)+)", re.IGNORECASE)


def split_sentences_with_separators(text: str) -> tuple[list[str], list[str]]:
    """
    Split text into sentences and the separators between them, such that
    ``separators[0] + sentences[0] + separators[1] + sentences[1] + ...``
    (with ``len(separators) == len(sentences) + 1``) reproduces `text` exactly.
    Separators (whitespace and <br /> tags) are kept so the sentence-based
    policies preserve the review's paragraph layout like the other policies.
    """
    parts = _SENTENCE_SEPARATOR_RE.split(text)
    sentences: list[str] = []
    separators: list[str] = [""]
    for i, part in enumerate(parts):
        if i % 2 == 1:  # separator
            separators[-1] += part
        elif part:      # non-empty sentence
            sentences.append(part)
            separators.append("")
    return sentences, separators


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' + whitespace, or <br /> breaks."""
    return split_sentences_with_separators(text)[0]


def deletion_count(num_tokens: int, n: float) -> int:
    """Number of tokens every policy removes from a `num_tokens`-token review at
    level `n`: round(n * num_tokens), always leaving at least one token."""
    if num_tokens <= 1:
        return 0
    return min(num_tokens - 1, int(round(num_tokens * n)))


# --------------------------------------------------------------------------- #
# Deletion policies (operate on tokens; return plain detokenized text)
# --------------------------------------------------------------------------- #

def random_deletion(text: str, n: float) -> str:
    """
    Random Deletion: remove a random selection of exactly round(n * len(tokens))
    tokens (sampled without replacement), rather than flipping an independent
    coin per token. This makes the removed fraction deterministic instead of
    binomially distributed, matching the other policies' exactness.
    """
    tokens = tokenize(text)
    num_tokens = len(tokens)
    num_to_remove = deletion_count(num_tokens, n)
    if num_to_remove <= 0:
        return text

    remove_indices = set(np.random.choice(num_tokens, size=num_to_remove, replace=False))
    kept = [t for i, t in enumerate(tokens) if i not in remove_indices]
    return detokenize(kept)


def frontal_deletion(text: str, n: float) -> str:
    """Frontal Deletion: drop the front n fraction of tokens; keep the tail."""
    tokens = tokenize(text)
    drop_len = deletion_count(len(tokens), n)
    if drop_len <= 0:
        return text
    return detokenize(tokens[drop_len:])


def rear_deletion(text: str, n: float) -> str:
    """Rear Deletion: drop the back n fraction of tokens; keep the front."""
    tokens = tokenize(text)
    drop_len = deletion_count(len(tokens), n)
    if drop_len <= 0:
        return text
    return detokenize(tokens[: len(tokens) - drop_len])


def allocate_sentence_deletions(sentence_lengths: list[int], total: int) -> list[int]:
    """
    Split a deletion budget of `total` tokens across sentences in proportion to
    their lengths (largest-remainder rounding), so each sentence loses as close
    to the same fraction as possible while the review as a whole loses exactly
    `total` tokens. Flooring per sentence instead would systematically delete
    less than the target from reviews with many short sentences.
    """
    num_tokens = sum(sentence_lengths)
    if total <= 0 or num_tokens == 0:
        return [0] * len(sentence_lengths)
    total = min(total, num_tokens - 1)

    quotas = [length * total / num_tokens for length in sentence_lengths]
    counts = [int(q) for q in quotas]
    remainder = total - sum(counts)
    # Hand the leftover tokens to the sentences with the largest fractional
    # parts (ties broken by position, for determinism).
    order = sorted(range(len(quotas)), key=lambda i: (counts[i] - quotas[i], i))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _sentence_deletion(text: str, n: float, from_front: bool) -> str:
    """Shared implementation of the two sentence-based policies."""
    sentences, separators = split_sentences_with_separators(text)
    if not sentences:
        return text

    sentence_tokens = [tokenize(s) for s in sentences]
    # Budget is based on the whole review's token count, like every other
    # policy; only the sentence text itself is eligible for deletion.
    budget = deletion_count(len(tokenize(text)), n)
    drops = allocate_sentence_deletions([len(t) for t in sentence_tokens], budget)

    pieces = [separators[0]]
    for tokens, drop, separator in zip(sentence_tokens, drops, separators[1:]):
        kept = tokens[drop:] if from_front else tokens[: len(tokens) - drop]
        if kept:  # a sentence deleted entirely also drops its trailing separator
            piece = detokenize(kept)
            if not pieces[-1] or pieces[-1][-1].isspace():
                # A sentence's tail starts with a space-prefixed token (" waste");
                # after a whitespace separator that would leave a double space,
                # which tokenizes as an extra token.
                piece = piece.lstrip()
            pieces.append(piece)
            pieces.append(separator)
    return "".join(pieces).strip() or text


def sentence_frontal_deletion(text: str, n: float) -> str:
    """
    Sentence-based Frontal Deletion: split into sentences on punctuation and
    <br /> line breaks, then drop the front n fraction of EACH sentence's tokens. Deletion is
    applied per-sentence (rather than to the whole review at once) so that
    roughly n fraction of every sentence is removed, not just the opening
    sentence(s). The review as a whole loses the same number of tokens as
    under the other policies (see allocate_sentence_deletions).
    """
    return _sentence_deletion(text, n, from_front=True)


def sentence_rear_deletion(text: str, n: float) -> str:
    """Sentence-based Rear Deletion: like sentence_frontal_deletion, but
    drops the back n fraction of each sentence's tokens instead of the front."""
    return _sentence_deletion(text, n, from_front=False)


# Tokens. Chosen by exploring the v1 run's records, i.e. on the same 500 reviews
# it is evaluated on, so its accuracy here is in-sample (see README).
LENGTH_ADAPTIVE_THRESHOLD = 180


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

    # Reseed so random_deletion's choices depend only on (level, review), not
    # on which other policies happened to run earlier in this session.
    np.random.seed(RANDOM_SEED)
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


def evaluate(
    sample: pd.DataFrame,
    classifier: SentimentClassifier,
    tracker: ProgressTracker | None = None,
    context: str = "",
    log_every: int = PROGRESS_LOG_EVERY,
    orig_lengths: pd.Series | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Classify every row of `sample` (the already-degraded review sample) and
    return (accuracy, records). Unparsable model outputs are excluded from
    the accuracy and recorded with pred_label=-1 and correct=NaN.

    `records` is a DataFrame with one row per sampled review: its original
    index, true/predicted label, whether the prediction was correct, and
    token lengths (original, pre-degradation, and degraded/as-classified).
    `orig_lengths`, if given, should be a Series mapping df's index to the
    token length of the UNDEGRADED review (see compute_token_lengths).
    """
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

    accuracy = accuracy_score(truths, preds) if preds else float("nan")
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


def completed_blocks(records: pd.DataFrame | None) -> set[tuple[float, str]]:
    """(level, policy) pairs that already have a full sample's worth of records."""
    if records is None or records.empty:
        return set()
    counts = records.groupby(["level", "policy"]).size()
    return {(round(float(level), 1), policy) for (level, policy), c in counts.items() if c >= SAMPLE_SIZE}


def run_experiment(
    sample_df: pd.DataFrame,
    classifier: SentimentClassifier,
    orig_lengths: pd.Series,
    existing_records: pd.DataFrame | None = None,
    on_block_done: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """
    For each level n in NOISE_LEVELS and each policy in DELETION_METHODS,
    degrade every review in `sample_df` and classify it.

    (level, policy) blocks already present in `existing_records` are skipped,
    so a run that crashed or was interrupted resumes where it stopped, and a
    newly added policy only costs API calls for that policy. After each block,
    the combined records so far are passed to `on_block_done` (used to
    checkpoint data/run_records.csv).

    Returns a DataFrame with one row per individual classification (existing
    plus new), used for the plots, the raw-data export, and to build the
    aggregate ExperimentResults via build_experiment_results().
    """
    done = completed_blocks(existing_records)
    todo = [(n, key) for n in NOISE_LEVELS for key in DELETION_METHODS if (round(n, 1), key) not in done]
    all_records: list[pd.DataFrame] = []
    if existing_records is not None and not existing_records.empty:
        all_records.append(existing_records)

    if not todo:
        logger.info("All level/policy blocks already recorded -- skipping API calls entirely")
        return pd.concat(all_records, ignore_index=True)

    total_calls_estimate = len(todo) * len(sample_df)
    tracker = ProgressTracker(total_calls_estimate)
    classifier.on_call = tracker.record_call  # wire real (non-cached) API calls into the tracker
    logger.info(
        "Starting experiment: %d of %d level/policy blocks to run x %d samples "
        "(up to %d API calls, fewer if caching kicks in)",
        len(todo),
        len(NOISE_LEVELS) * len(DELETION_METHODS),
        len(sample_df),
        total_calls_estimate,
    )

    for block_idx, (n, method_key) in enumerate(todo, start=1):
        label = POLICY_LABELS[method_key]
        logger.info("--- %s at level %.1f (block %d/%d) ---", label, n, block_idx, len(todo))

        degraded_df = apply_deletion(sample_df, n, method_key)
        acc, records_df = evaluate(
            degraded_df,
            classifier,
            tracker=tracker,
            context=f"level={n:.1f} | {label}",
            orig_lengths=orig_lengths,
        )
        records_df["level"] = n
        records_df["policy"] = method_key
        records_df["policy_label"] = label
        records_df["version"] = EXPERIMENT_VERSION
        all_records.append(records_df)

        logger.info("%s @ level %.1f -> accuracy=%.4f", label, n, acc)
        if on_block_done:
            on_block_done(pd.concat(all_records, ignore_index=True))

    logger.info("Experiment complete in %s", format_duration(time.monotonic() - tracker.start_time))
    return pd.concat(all_records, ignore_index=True)


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
    client = OpenAI(api_key=api_key, max_retries=MAX_API_RETRIES) if api_key else None

    logger.info("Loading dataset")
    dataset = load_dataset("stanfordnlp/imdb")
    test_df = pd.DataFrame(dataset["test"])

    logger.info("Computing original review token lengths")
    orig_lengths = compute_token_lengths(test_df["text"])
    write_dataset_summary(test_df, orig_lengths)

    # One fixed sample of reviews, shared by every level and policy.
    sample_df = test_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Resume from data/run_records.csv if a previous run (or an interrupted
    # one) already recorded some level/policy blocks under this version.
    run_records_path = os.path.join(DATA_DIR, "run_records.csv")
    existing_records: pd.DataFrame | None = None
    if os.path.exists(run_records_path):
        existing_records = pd.read_csv(run_records_path)
        versions = set(existing_records.get("version", pd.Series([1])).fillna(1).astype(int))
        if versions != {EXPERIMENT_VERSION}:
            raise RuntimeError(
                f"{run_records_path} holds records from experiment version(s) {sorted(versions)}, "
                f"but this code is version {EXPERIMENT_VERSION}. Move or delete that file so "
                "results from different code versions are not mixed."
            )
        if set(existing_records["index"]) != set(sample_df.index):
            raise RuntimeError(f"{run_records_path} was recorded on a different review sample.")
        logger.info("Found existing %s -- resuming from its recorded blocks", run_records_path)

    todo = len(NOISE_LEVELS) * len(DELETION_METHODS) - len(completed_blocks(existing_records))
    if todo and client is None:
        raise RuntimeError("Set OPENAI_API_KEY in .env (see .env.example) before running.")

    master_records = run_experiment(
        sample_df,
        SentimentClassifier(client),
        orig_lengths,
        existing_records=existing_records,
        on_block_done=save_run_data,  # checkpoint after every block
    )

    results = build_experiment_results(master_records)

    plot_results(results)
    write_results_txt(results)
    save_run_data(master_records)

    length_bin_edges = compute_length_bin_edges(orig_lengths)
    for n in NOISE_LEVELS:
        plot_length_vs_accuracy(master_records, n, length_bin_edges)


if __name__ == "__main__":
    main()