# Positional Token Deletion and LLM Sentiment Classification

How does deleting parts of a text affect an LLM's ability to recover its sentiment, and does *where* the text is deleted matter?

This project degrades IMDb movie reviews with several position-based token deletion policies at increasing severity, then asks GPT-4.1 mini to classify each degraded review as Positive or Negative. The goal is to understand how sentence structure and token position carry sentiment, and to design a deletion policy that preserves it best.

The paper draft is in [`paper/research_draft.pdf`](paper/research_draft.pdf). **Status: work in progress.** The optimal-policy section (RQ3) and the discussion are not finished yet.

## Research questions

- **RQ1:** How does LLM accuracy change as deletion severity increases?
- **RQ2:** Does the method of deletion have significantly different effects on accuracy?
- **RQ3:** Can a deletion policy be designed that preserves the most information and gives consistently higher accuracy than the other methods?

## Deletion policies

All deletion operates on GPT tokens (via `tiktoken`). Severity `n` is the fraction of tokens removed, swept from 0.0 to 0.8 in steps of 0.1.

| Policy | What is deleted |
|---|---|
| Random | A random `n` fraction of tokens (baseline) |
| Frontal | The first `n` fraction of the review |
| Rear | The last `n` fraction of the review |
| Sentence-based Frontal | The first `n` fraction of **each sentence** |
| Sentence-based Rear | The last `n` fraction of **each sentence** |
| Length-Adaptive *(experimental)* | Rear deletion for reviews of ≤180 tokens, Frontal for longer ones |

## Results

There are 500 reviews (the same 500 at every policy/severity combination), sampled from the IMDb test split with seed 42. Each cell shows accuracy.

| Severity | Random | Frontal | Rear | SB Frontal | SB Rear | Length-Adaptive* |
|---|---|---|---|---|---|---|
| 0.0 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 | 0.982 |
| 0.1 | 0.968 | **0.980** | 0.966 | 0.972 | 0.962 | 0.974 |
| 0.2 | 0.950 | 0.970 | 0.962 | 0.962 | 0.962 | **0.972** |
| 0.3 | 0.930 | 0.966 | 0.960 | 0.962 | 0.942 | **0.968** |
| 0.4 | 0.858 | 0.956 | 0.954 | 0.940 | 0.918 | **0.968** |
| 0.5 | 0.812 | 0.948 | 0.924 | 0.924 | 0.906 | **0.952** |
| 0.6 | 0.716 | 0.936 | 0.920 | 0.884 | 0.874 | **0.940** |
| 0.7 | 0.634 | **0.922** | 0.890 | 0.854 | 0.770 | 0.912 |
| 0.8 | 0.590 | **0.888** | 0.838 | 0.810 | 0.666 | 0.868 |

\*Length-Adaptive is an experimental candidate for RQ3. Its 180-token threshold was chosen by exploring these same 500 reviews, so these numbers are in-sample. It still needs to be evaluated on held-out reviews.

![Accuracy vs. deletion level](results/results.png)

Per-severity plots of accuracy against original review length are in [`results/plots/`](results/plots/).

## Repository layout

```
├── run_experiment.py        # full pipeline: deletion policies, LLM classification, plots
├── requirements.txt
├── paper/
│   └── research_draft.pdf   # current paper draft
├── data/
│   ├── run_records.csv      # every individual classification (27,000 rows)
│   └── dataset_summary.txt  # IMDb test-set class balance and token-length stats
└── results/
    ├── results.txt          # accuracy table (all policies × severities)
    ├── results.png          # accuracy vs. severity plot
    └── plots/               # accuracy vs. review length, one plot per severity
```

`data/run_records.csv` columns: `index` (row in the IMDb test split), `true_label`, `pred_label` (1 = positive, 0 = negative, -1 = unparsable), `correct`, `orig_length` and `degraded_length` (in tokens), `level`, `policy`, and `policy_label`.

## Reproducing

```bash
pip install -r requirements.txt
python run_experiment.py
```

The script reuses `data/run_records.csv`. It only calls the API for policies not already recorded there, using the same review sample. With the committed data, every policy is already present, so the command regenerates all tables and plots **without an API key or any API calls**.

To run a new policy, add it to `DELETION_METHODS` / `POLICY_LABELS` and set your key first:

```bash
export OPENAI_API_KEY=...   # never commit this
python run_experiment.py
```

To rerun everything from scratch, move `data/run_records.csv` aside first. That run makes about 27,000 API calls. Note that the model is called at its default temperature, so a rerun will not reproduce the numbers exactly.

## Related work

- Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*: the IMDb dataset
- Jin et al. (2020), *Is BERT Really Robust?* (TextFooler)
- Jiang et al. (2023), *LLMLingua*; Pan et al. (2024), *LLMLingua-2*; Jiang et al. (2024), *LongLLMLingua*
- Herrera et al., prompt-level noise and input-cleaning for LLM inference
