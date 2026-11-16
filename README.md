# Crawl_Counts

Reproducibility companion for **What a Crawl Counts: Auditing Counting-Unit Sensitivity in Web Corpora**.

## Quick start

```sh
python -m pip install -r requirements-core.txt
python reproduce_core.py
```

The offline entry checks payload hashes and reconstructs the 77-cell training/evaluation crossing, 12 control values, 900 supplementary error cells, four paired H intervals, and reference coverage. It uses frozen per-record laws, labels and predictions and starts no fitting.

Read [the core report](report/REPORT.md) and [all native-topic ranges](report/all_topic_ranges.csv). All six topics are reported for each declared candidate set, including compatible ranges. Health is an exploratory illustration. The .50/.70 sets have minimum health-share shortfalls of about 2.85/2.68 percentage points over every allowed convex mixture.

## Rebuild the coverage panels

```sh
python -m pip install -r requirements-plot.txt
python reproduce_core.py --coverage-only --plot
```

Generated reports and standalone Figure 2(a,b) PNG/SVG panels go under `recomputed/`. `report/` contains the frozen report snapshot.

## Contents

- `cc_training_weights.npz`: eleven full-corpus training-weight definitions and record indices, before the registered normalization.
- `cc_evaluation.npz`, `bbc_evaluation.npz`: laws restricted and renormalized on fixed test records.
- Prediction arrays: saved per-seed results for the original crossing, controls and extensions.
- `matched_inputs.npz`: both models and both laws on shared IDs for 5,000 joint test-cluster resamples per H setting.
- `content_laws.npz`, `reference_evaluation.npz`: 3Sources candidate laws and supplied story/topic reference.
- `source/`: original implementations for method inspection. Full training additionally requires the raw-source releases and original project configuration; the portable numeric entry is self-contained.
- `editable_tables/`: native LaTeX tables and numeric styles.

Full environment pins are in `requirements.txt`; `requirements-core.txt` lists the smaller numeric-entry dependency set. Source notices and fingerprints identify the public research releases and usage conditions. Raw news text and model parameters are excluded.

## Interpretation and access

Exact candidate ranges, center-radius bands and paired bootstrap intervals describe different objects. A zero scalar gap permits that individual mean; it does not establish joint attainability of the six-topic vector. H intervals are post-hoc conditional descriptions with models, partitions and splits fixed.

This is the author's public GitHub repository. The double-blind manuscript uses its portable artifact entry; an anonymous reviewer link requires a separately anonymized delivery channel.
