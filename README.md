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

> **These are the v1 results from the paper draft, produced by the original code.** That code had three known problems:
> - The model ran at its default temperature, so results varied from run to run.
> - Sentences that met at a `<br />` paragraph break were treated as one sentence (this affected 52% of the reviews).
> - The sentence-based policies did not delete the stated fraction: about 5 points too little for SB Frontal and about 2 too much for SB Rear.
>
> The code is now fixed (v2). The v1 data is kept unchanged in [`archive/v1/`](archive/v1/). Rerun the experiment to produce v2 results (see [Reproducing](#reproducing)).

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

![Accuracy vs. deletion level (v1)](archive/v1/results/results.png)

Per-severity plots of accuracy against original review length are in [`archive/v1/results/plots/`](archive/v1/results/plots/).

## Repository layout

```
├── run_experiment.py        # full pipeline: deletion policies, LLM classification, plots
├── requirements.txt
├── .env.example             # template for your local .env (API key)
├── paper/
│   └── research_draft.pdf   # current paper draft
├── data/
│   ├── dataset_summary.txt  # IMDb test-set class balance and token-length stats
│   └── run_records.csv      # every individual classification (created by a run)
├── results/                 # created by a run
│   ├── results.txt          # accuracy table (all policies × severities)
│   ├── results.png          # accuracy vs. severity plot
│   └── plots/               # accuracy vs. review length, one plot per severity
└── archive/v1/              # original run behind the paper draft's numbers
    ├── run_records.csv      # 27,000 rows
    └── results/
```

`run_records.csv` columns: `index` (row in the IMDb test split), `true_label`, `pred_label` (1 = positive, 0 = negative, -1 = unparsable), `correct`, `orig_length` and `degraded_length` (in tokens), `level`, `policy`, `policy_label`, and `version` (v2 only).

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY=sk-... in .env (gitignored, never committed)
python run_experiment.py
```

A full run makes up to 27,000 calls to `gpt-4.1-mini` at temperature 0. Identical inputs are cached, so there are somewhat fewer actual calls. The run saves `data/run_records.csv` after every level/policy block. If it is interrupted, run the same command again and it will continue where it stopped. Once every block is recorded, rerunning only regenerates the tables and plots, with no API calls.

To add a policy, add it to `DELETION_METHODS` / `POLICY_LABELS` and rerun. Only the new blocks call the API. If you change anything that alters the text sent to the model, bump `EXPERIMENT_VERSION` and move the old `data/run_records.csv` aside. The script will refuse to mix records from different versions.

All policies remove `round(n × review tokens)` tokens, so every policy is compared at the same deletion budget.

## Related work

- Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*: the IMDb dataset
- Jin et al. (2020), *Is BERT Really Robust?* (TextFooler)
- Jiang et al. (2023), *LLMLingua*; Pan et al. (2024), *LLMLingua-2*; Jiang et al. (2024), *LongLLMLingua*
- Herrera et al., prompt-level noise and input-cleaning for LLM inference
