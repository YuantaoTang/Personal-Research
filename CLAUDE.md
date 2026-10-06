# Project context for Claude Code

Research project: **how position-based token deletion affects LLM (gpt-4.1-mini) sentiment classification of IMDb reviews**, and designing an optimal deletion policy (RQ3). Paper draft: `paper/research_draft.pdf` (work in progress; sections 5.1 "Optimal Deletion Policy" and Discussion unfinished). Private GitHub repo: https://github.com/YuantaoTang/Personal-Research (branch `main`).

## Files
- `policies.py`: tokenization (tiktoken) and every deletion policy as a **priority function**. Each token gets a score, and at level `n` the `round(n·N)` lowest-scoring tokens are deleted. Baselines: random, frontal, rear, sentence_frontal, sentence_rear, length_adaptive (Rear if ≤180 tokens, else Frontal). `PriorityModel` is the learned policy: `w = f(x) + g(s) + z·h(x)`, where x = position in review, s = position in sentence, z = standardized log length, and f, g, h are Bernstein polynomials.
- `run_experiment.py`: degrades reviews, classifies them via the OpenAI API, and writes records, tables and plots. Options:
  - `--split train|holdout`: train is the original 500 reviews (seed 42, IMDb **test** split); holdout is 500 disjoint reviews and adds the learned policy.
  - `--workers N`: concurrent requests, default 8.
  - Behaviour: checkpoints after every level/policy block and resumes on rerun. Records carry `EXPERIMENT_VERSION = 3`, and mixing versions is refused. Each record stores a hex `kept_mask` of which original tokens were kept.
- `optimize_policy.py`:
  - `fit`: L2 logistic regression of `correct` on kept-token basis features; degrees and C chosen by 5-fold CV grouped by review; compared against a base-rate model and a position-blind model. Writes `results/learned_policy/{policy.json, priority_function.png, fit_report.txt}`. It refuses incomplete train data.
  - `evaluate`: paired exact McNemar tests of the learned policy vs each baseline on holdout, written to `results/holdout/comparison.txt`.
- `archive/v1/`: the original run (buggy old code) behind the paper draft's results table. Keep it unchanged.
- API key: `OPENAI_API_KEY` in `.env` (gitignored; template `.env.example`). Never commit keys.

## Workflow
```
python run_experiment.py                  # 1. train split (6 baselines × 9 levels × 500 = 27,000 calls)
python optimize_policy.py fit             # 2. fit the learned priority function (no API calls)
python run_experiment.py --split holdout  # 3. holdout (baselines + learned_<hash>)
python optimize_policy.py evaluate        # 4. significance tests (no API calls)
```
Refitting creates a new `learned_<hash>` policy, so step 3 then only runs that policy's 4,500 calls.

## Current status (2026-10-05)
- **Train run is incomplete: 11 of 54 blocks** in `data/train_records.csv`. It was interrupted and needs resuming with `python run_experiment.py`.
- **Holdout run is complete** (63 blocks), but its learned policy `learned_9ceac7e9` was fitted on the partial train data (levels 0.0–0.1 only) and **is invalid**. Its CV loss was worse than the position-blind model. Ignore its results, and rerun step 2 then step 3 once train is complete.
- `data/*_records.csv` and `results/` are **not committed**. They cost API money to regenerate, so consider committing the CSVs.

## Findings so far (holdout baselines are valid out-of-sample data)
- **Accuracy at level 0.8:** Frontal 0.872, Length-Adaptive 0.864, Rear 0.842, Sentence-based Frontal 0.664, Sentence-based Rear 0.654, Random 0.568.
- **Frontal is the policy to beat.** Length-Adaptive is not better than Frontal out-of-sample.
- **Fragmentation dominates.** Accuracy ranks by the number of separate deleted spans: one contiguous hole beats many small ones. A model using only "spans deleted" plus level and length almost matches a full policy×level model (AIC 12765 vs 12740).
- **Limitation of the current learned model:** it values each token independently of its neighbours, so it can't capture fragmentation. Its sentence-position term scatters deletions, which is why the invalid fit did badly.

## Suggested next steps (not yet implemented; the user had not decided)
1. Finish the train run, then refit and run the holdout step.
2. Make the learned policy fragmentation-aware. Preferred option: a **contiguous-window policy** that deletes one block placed where the learned importance is lowest (Frontal and Rear are special cases). Alternative: add a "deleted spans per token" feature to the regression.
3. Add probe policies (delete a centred block; keep only the middle) so the regression sees middle deletions. About 9,000 extra calls.
4. Cross-fitting: fit on train and test on holdout, then swap, for more statistical power.
5. Optional, outside the paper's position-only scope: token-content features (stopwords, punctuation, HTML tags).

## Paper corrections still needed (in the PDF; not done)
- Section 4.2 says reviews come from the *training* set, but the code uses the *test* split.
- The definition of Sentence-based Rear says "back of the prompt"; it should say "back of each sentence".
- "XXX" placeholder in section 4.2 (now: 500 holdout reviews).
- "Mass" → "Maas"; "et. al." → "et al."
- Duplicate section "5.1"; Discussion should be section 6.
- H1.1 ("strictly decrease") was contradicted in v1 (Frontal rose from 0.976 to 0.980 at level 0.1).
- The results table must be replaced with v3 numbers.
- The old code's flaws (default temperature, `<br />` sentence merging, sentence-policy deletion budgets) are documented in the README.

## Notes
- The user's old OpenAI key was hardcoded in the original script and was removed before the first commit. The user was told to revoke it.
- No license has been chosen; the repo is private.
- Commit messages end with a `Co-Authored-By: Claude` line.
