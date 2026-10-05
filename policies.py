"""
Token-level deletion policies, all expressed as priority functions.

A priority function maps a tokenized review to one score per token. At
deletion level n, the round(n * N) lowest-scoring tokens of an N-token review
are deleted (ties broken by position, earlier first) and the remaining tokens
are detokenized back into text. So every policy removes exactly the same
number of tokens from a given review at a given level, and a policy is fully
described by the order in which it deletes tokens.

The baseline policies from the paper are simple functions of token position.
The learned policy (PriorityModel) is a continuous function w(token features)
fitted by optimize_policy.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import tiktoken

MODEL_NAME = "gpt-4.1-mini"
RANDOM_SEED = 42

# Tokens. Chosen by exploring the v1 run's records (the same 500 reviews it was
# evaluated on), so only its holdout-split accuracy is an unbiased estimate.
LENGTH_ADAPTIVE_THRESHOLD = 180


# --------------------------------------------------------------------------- #
# Tokenization
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


def tokenize_batch(texts: list[str]) -> list[list[int]]:
    """Tokenize many texts at once (multithreaded where tiktoken supports it)."""
    try:
        return _ENCODING.encode_ordinary_batch(texts, num_threads=8)
    except AttributeError:
        # Older tiktoken versions without batch encoding: fall back to a loop.
        return [tokenize(t) for t in texts]


def detokenize(tokens: list[int]) -> str:
    """List of GPT token ids -> text. Reverses `tokenize`."""
    return _ENCODING.decode(tokens)


# A sentence ends at whitespace after '.', '!' or '?', or at a run of HTML line
# breaks. IMDB reviews mark paragraphs with "<br /><br />", often directly after
# the punctuation ("...end.<br /><br />Next..."), so splitting on punctuation +
# whitespace alone would merge sentences across paragraphs.
_SENTENCE_SEPARATOR_RE = re.compile(r"(?<=[.!?])\s+|\s*(?:<br\s*/?>\s*)+", re.IGNORECASE)


def sentence_start_offsets(text: str) -> np.ndarray:
    """Character offsets at which each sentence starts. The separator after a
    sentence (whitespace, <br /> tags) counts as part of that sentence."""
    starts = [0] + [m.end() for m in _SENTENCE_SEPARATOR_RE.finditer(text) if 0 < m.end() < len(text)]
    return np.unique(starts)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (each including its trailing separator)."""
    bounds = list(sentence_start_offsets(text)) + [len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


# --------------------------------------------------------------------------- #
# Tokenized review with per-token positional features
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class ReviewTokens:
    """A review's tokens plus the per-token features priority functions use."""

    review_id: int
    text: str
    tokens: list[int]
    position: np.ndarray           # token index / (N - 1): 0 = first token, 1 = last
    sentence_position: np.ndarray  # same, but within the token's own sentence

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


def prepare_review(review_id: int, text: str) -> ReviewTokens:
    """Tokenize a review once and compute each token's positional features."""
    tokens = tokenize(text)
    num = len(tokens)
    position = np.arange(num) / max(num - 1, 1)

    sentence_position = np.zeros(num)
    if num:
        # Assign each token to the sentence its first character falls in.
        _, offsets = _ENCODING.decode_with_offsets(tokens)
        sentence = np.searchsorted(sentence_start_offsets(text), offsets, side="right") - 1
        first = np.r_[0, np.flatnonzero(np.diff(sentence)) + 1]  # first token of each sentence
        lengths = np.diff(np.r_[first, num])
        rank = np.arange(num) - np.repeat(first, lengths)
        sentence_position = rank / np.maximum(np.repeat(lengths, lengths) - 1, 1)

    return ReviewTokens(int(review_id), text, tokens, position, sentence_position)


# --------------------------------------------------------------------------- #
# Deletion by priority
# --------------------------------------------------------------------------- #

PriorityFn = Callable[[ReviewTokens], np.ndarray]


def deletion_count(num_tokens: int, n: float) -> int:
    """Number of tokens every policy removes from a `num_tokens`-token review at
    level `n`: round(n * num_tokens), always leaving at least one token."""
    if num_tokens <= 1:
        return 0
    return min(num_tokens - 1, int(round(num_tokens * n)))


def deletion_mask(scores: np.ndarray, n: float) -> np.ndarray:
    """Keep-mask that deletes the deletion_count() lowest-scoring tokens
    (ties broken by position, earlier tokens first)."""
    num = len(scores)
    keep = np.ones(num, dtype=bool)
    k = deletion_count(num, n)
    if k > 0:
        order = np.lexsort((np.arange(num), scores))  # primary key: scores
        keep[order[:k]] = False
    return keep


def apply_policy(review: ReviewTokens, n: float, priority: PriorityFn) -> tuple[str, np.ndarray]:
    """Degrade `review` at level `n` under `priority`. Returns (text, keep_mask)."""
    keep = deletion_mask(np.asarray(priority(review), dtype=float), n)
    if keep.all():
        return review.text, keep
    return detokenize([t for t, k in zip(review.tokens, keep) if k]), keep


def encode_mask(keep: np.ndarray) -> str:
    """Compact hex encoding of a keep-mask, for storing in the records CSV."""
    return np.packbits(keep).tobytes().hex()


def decode_mask(hex_mask: str, num_tokens: int) -> np.ndarray:
    """Inverse of encode_mask."""
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hex_mask), dtype=np.uint8))
    return bits[:num_tokens].astype(bool)


# --------------------------------------------------------------------------- #
# Baseline policies (the paper's five, plus the length-adaptive heuristic)
# --------------------------------------------------------------------------- #

def random_priority(review: ReviewTokens) -> np.ndarray:
    """A fixed random order per review, so deletions are nested across levels."""
    return np.random.default_rng([RANDOM_SEED, review.review_id]).random(review.num_tokens)


def frontal_priority(review: ReviewTokens) -> np.ndarray:
    """Front of the review is deleted first."""
    return review.position


def rear_priority(review: ReviewTokens) -> np.ndarray:
    """Back of the review is deleted first."""
    return -review.position


def sentence_frontal_priority(review: ReviewTokens) -> np.ndarray:
    """Front of every sentence is deleted first, so at level n roughly the
    front n fraction of EACH sentence is removed."""
    return review.sentence_position


def sentence_rear_priority(review: ReviewTokens) -> np.ndarray:
    """Back of every sentence is deleted first."""
    return -review.sentence_position


def length_adaptive_priority(review: ReviewTokens) -> np.ndarray:
    """Rear deletion for short reviews (verdict tends to come early), frontal
    deletion for long ones (verdict tends to come after a synopsis)."""
    if review.num_tokens <= LENGTH_ADAPTIVE_THRESHOLD:
        return rear_priority(review)
    return frontal_priority(review)


BASELINE_POLICIES: dict[str, tuple[str, PriorityFn]] = {
    "random": ("Random Deletion", random_priority),
    "frontal": ("Frontal Deletion", frontal_priority),
    "rear": ("Rear Deletion", rear_priority),
    "sentence_frontal": ("Sentence-based Frontal Deletion", sentence_frontal_priority),
    "sentence_rear": ("Sentence-based Rear Deletion", sentence_rear_priority),
    "length_adaptive": ("Length-Adaptive Deletion", length_adaptive_priority),
}


# --------------------------------------------------------------------------- #
# Learned policy: a continuous priority function fitted by regression
# --------------------------------------------------------------------------- #

def bernstein_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Bernstein polynomial basis on [0, 1]: one column per basis function.
    Any smooth curve on [0, 1] is approximated by a weighted sum of these."""
    k = np.arange(degree + 1)
    binom = np.array([math.comb(degree, i) for i in k], dtype=float)
    x = np.asarray(x, dtype=float)[:, None]
    return binom * x**k * (1 - x) ** (degree - k)


@dataclass
class PriorityModel:
    """
    Token importance as a continuous function of where the token sits:

        w(token) = f(x) + g(s) + z * h(x)

    x = position in the review (0..1), s = position within the sentence (0..1),
    z = standardized log review length. f, g and h are polynomials in
    Bernstein form, so w is smooth, and the review-length interaction lets
    short and long reviews prefer different deletion orders. Used as a
    priority function: the lowest-w tokens are deleted first.
    """

    degree_position: int
    degree_sentence: int
    log_length_mean: float
    log_length_std: float
    coef_position: list[float] | None = None
    coef_sentence: list[float] | None = None
    coef_position_x_length: list[float] | None = None

    @property
    def num_features(self) -> int:
        return 2 * (self.degree_position + 1) + self.degree_sentence + 1

    def length_z(self, num_tokens: int) -> float:
        return (math.log(max(num_tokens, 1)) - self.log_length_mean) / self.log_length_std

    def basis(self, position: np.ndarray, sentence_position: np.ndarray, num_tokens: int) -> np.ndarray:
        """Per-token basis matrix (tokens x num_features)."""
        bx = bernstein_basis(position, self.degree_position)
        bs = bernstein_basis(sentence_position, self.degree_sentence)
        return np.hstack([bx, bs, bx * self.length_z(num_tokens)])

    def token_basis(self, review: ReviewTokens) -> np.ndarray:
        return self.basis(review.position, review.sentence_position, review.num_tokens)

    @property
    def coefficients(self) -> np.ndarray:
        return np.r_[self.coef_position, self.coef_sentence, self.coef_position_x_length]

    def with_coefficients(self, coef: np.ndarray) -> "PriorityModel":
        coef = [float(c) for c in coef]
        a, b = self.degree_position + 1, self.degree_position + self.degree_sentence + 2
        return PriorityModel(
            self.degree_position,
            self.degree_sentence,
            self.log_length_mean,
            self.log_length_std,
            coef[:a],
            coef[a:b],
            coef[b:],
        )

    def __call__(self, review: ReviewTokens) -> np.ndarray:
        return self.token_basis(review) @ self.coefficients

    def fingerprint(self) -> str:
        """Short id that changes whenever the fitted function changes."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:8]

    def save(self, path: str, metadata: dict | None = None) -> None:
        with open(path, "w") as f:
            json.dump({"model": asdict(self), "fingerprint": self.fingerprint(), "fit": metadata or {}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PriorityModel":
        with open(path) as f:
            return cls(**json.load(f)["model"])
