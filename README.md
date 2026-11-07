# Reproducing and Extending SimCSE

COMP8240 Applications of Data Science — Macquarie University, Session 2, 2026
Author: Trilok Kumar Reddy Ande

This repository contains the code and data for my project replicating
**SimCSE: Simple Contrastive Learning of Sentence Embeddings**
(Gao, Yao & Chen, EMNLP 2021 — [paper](https://aclanthology.org/2021.emnlp-main.552/), [official code](https://github.com/princeton-nlp/SimCSE)),
and testing it on two new datasets: **PAWS** and **WikiSim** (a new dataset built from Wikipedia).

## What is in this repository

| File | What it is |
|---|---|
| `SimCSE_replication.ipynb` | The full notebook: replication, PAWS, WikiSim, annotation, LLM judge |
| `extra_analysis.ipynb` | Extra checks for the final report: exact library versions, a second replication run, STS test-set sizes, bootstrap 95% confidence intervals, the WikiSim box plot and example errors. It only reads the saved WikiSim files |
| `extra_analysis_run_2026-11-06.ipynb` | The saved run of `extra_analysis.ipynb` (6 Nov 2026) with all text outputs: versions, replication, STS sizes, confidence intervals and error examples used in the report |
| `train_unsup.ipynb` | Our own re-training of unsupervised SimCSE on the authors' 1M Wikipedia sentences (paper settings, 54 min on a T4). Saved with outputs: average STS score 72.55 vs 76.25 in the paper |
| `train_unsup_checkpoints_run.ipynb` | Checkpoint test (saved run): re-training on 100k sentences, scoring the STS-B dev set every 125 steps. The dev score rose at every check (53.74 at step 125 to 68.29 at the end), so the best checkpoint was the final one (report Section 3.4) |
| `broken_pairs_check.ipynb` | Small CPU check (saved run): Spearman with the levels with and without the 6 broken-sentence pairs (unsup 70.13 / 70.62, sup 71.49 / 72.11) |
| `requirements.txt` | Exact package versions used in our final run |
| `data/raw_articles.json` | The 20 Wikipedia articles used (with revision IDs) |
| `data/sentences.json` | 6,098 cleaned sentences |
| `data/wikisim_pairs.csv` | The 400 WikiSim pairs with levels and SimCSE scores |
| `data/annotation_sheet.csv` | Blank sheet of the 100 pairs given to annotators (levels hidden) |
| `data/annotation_done.csv` | The author's 0–5 scores for the 100 pairs |
| `data/annotation_claude.csv` | Second annotation of the same 100 pairs by an **AI model (Claude, Anthropic)** — not a human annotator |
| `data/annotated_100.csv` | The 100 pairs with all scores (human, SimCSE, word overlap, LLM judge) |

`SimCSE_replication.ipynb` was last re-run on 7 Nov 2026 with PyTorch 2.14.1 / transformers 5.22.1; `requirements.txt` lists the versions of the original run.

## How to run

1. Open `SimCSE_replication.ipynb` in **Google Colab** and choose a **GPU runtime** (Runtime → Change runtime type → T4 GPU).
2. Upload the `data` folder next to the notebook (or clone this repo in Colab).
3. Run all cells from top to bottom (Runtime → Run all).

The notebook runs in this order:

| Section | What it does | Time on T4 |
|---|---|---|
| 0. Setup | Sets the data folder, prints library versions | seconds |
| 1. Replication | Clones SimCSE, fixes one line, evaluates both models on 7 STS tasks | ~3 min |
| 2. PAWS | Scores 8,000 PAWS test pairs, computes ROC-AUC | ~2 min |
| 3. WikiSim | Loads (or downloads) the articles, cleans them, builds 400 pairs, scores them | ~1 min |
| 4. Annotation | Compares SimCSE with the author's scores, and the author with the AI annotator | seconds |
| 5. LLM judge | Runs Qwen2.5-3B-Instruct as a judge on the 100 pairs (downloads ~6 GB) | ~5 min |

## Fix needed for the original code

The official `evaluation.py` calls `tokenizer.batch_encode_plus`, which was removed from recent
versions of `transformers` (the paper used 4.2.1). The notebook fixes it with:

```bash
sed -i "s/tokenizer.batch_encode_plus(/tokenizer(/g" evaluation.py
```

## Main results

**Replication (Spearman × 100, STS test sets):**

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg. |
|---|---|---|---|---|---|---|---|---|
| Unsup-SimCSE (ours = paper) | 68.40 | 82.41 | 74.38 | 80.91 | 78.56 | 76.85 | 72.23 | 76.25 |
| Sup-SimCSE (ours) | 75.30 | 84.67 | 80.19 | 85.40 | 80.82 | 84.26 | 80.39 | 81.58 |
| Sup-SimCSE (paper) | 75.30 | 84.67 | 80.19 | 85.40 | 80.82 | 84.25 | 80.39 | 81.57 |
| Unsup-SimCSE (our re-training) | 63.31 | 78.24 | 70.33 | 77.07 | 78.06 | 71.66 | 69.15 | 72.55 |

**New data:**

| Method | PAWS ROC-AUC | WikiSim Spearman (400 pairs) |
|---|---|---|
| Word overlap | 0.498 | 30.29 |
| Unsup-SimCSE | 0.585 | 70.13 |
| Sup-SimCSE | 0.647 | 71.49 |

## Data licences

- WikiSim is built from English Wikipedia text, licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The revision ID of every article is stored in `data/raw_articles.json`.
- PAWS is downloaded from [Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws) and is not redistributed here.
- The STS datasets are downloaded by the SimCSE/SentEval script and are not redistributed here.

## Use of AI

The second annotation (`annotation_claude.csv`) was produced by an AI model (Claude, Anthropic),
not by a human. In that file the score column is named `human_score` only to match the annotation-sheet format;
these scores are from the AI annotator. Generative AI was also used to help clean this code and draft the report.

## Note on re-running the LLM judge

`SimCSE_replication.ipynb` is saved with the outputs of a full re-run on 7 Nov 2026 (PyTorch 2.14.1, transformers 5.22.1).
All numbers match the report except LLM judge version 1, which gave Spearman 26.59 with the author's scores instead of 29.32
(score counts 42/48/2/8 instead of 39/51/2/8). Small numerical differences in half-precision GPU computation can change the
model's single-digit answer for a few pairs. LLM judge version 2 (62.50) and every other result were identical. The report mentions this.

## Note on STS12 and STS13 sizes

`extra_analysis.ipynb` counts every gold-standard file in the STS folders, so it prints 6,216 for STS12 and 2,250 for STS13.
The STS12 folder also contains a combined `ALL` file, and the STS13 folder contains an `SMT` subset, and SentEval does not use either.
The test sets actually used in the evaluation have 3,108 (STS12) and 1,500 (STS13) pairs.
