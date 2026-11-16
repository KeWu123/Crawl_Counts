# What a Crawl Counts

Code and frozen numeric inputs for counting-target coverage and matched training/evaluation audits.

## Quick start

Run `python -m pip install -r requirements-core.txt`, then `python reproduce_core.py`.
Read `report/REPORT.md` for all six topics. Optional Figure 2(a,b) reconstruction uses `requirements-plot.txt` and `python reproduce_core.py --coverage-only --plot`.

The source directory contains the original implementations; full training additionally needs the identified raw source releases. This repository contains the compact numerical reproduction inputs.

Review distribution uses a permitted anonymous snapshot; the author repository itself is private.

---

# Anonymous counting-target audit material

This current-version numeric artifact contains partitions/record IDs, all eleven full-corpus CC-News training-weight definitions, actual normalized test laws, individual-seed predictions, 60 supplementary-fit checkpoint predictions, controls, candidate mixture parameters and LP certificates. It contains no raw news text, model weights, user review history or author-identifying local paths.

Install `requirements.txt`, then run `python recompute.py`. The offline entry checks file hashes and reconstructs the 77-cell training/evaluation crossing, 12 displayed control values, 900 supplementary error cells, four joint paired-bootstrap H intervals, all 3Sources topic ranges/coverage and the original full-density CC-News radius. It starts no fitting. The partition and training weight arrays allow inspection of the population being counted.

`cc_training_weights.npz` stores inverse full-corpus group sizes before the registered minibatch/global normalization, together with train/test record indices. `cc_evaluation.npz` stores the full-corpus laws restricted and renormalized on the same tests. They are different stages. `matched_inputs.npz` preserves both models and both test laws on matched IDs. The zero BBC-linear refit change is specific to these saved predictions.

The new 3Sources rules align exact released stems, use positive log-TF / smooth IDF / L2 rows, and connect cosine >=.50 or >=.70. Neither story IDs nor topic labels enter construction or center optimization; the earlier reference baseline had already been seen. Supplied IDs provide a finite reference, not universal semantic truth. LP centers are convex candidate mixtures, not arbitrary free distributions. H intervals are post-hoc conditional percentile descriptions on fixed models/splits; the figure's small resample cloud is not a 95% contour. Exact candidate statistic ranges, TV-radius bands and bootstrap intervals are distinct objects.

Numeric reproduction does not repeat training or demonstrate raw-data semantic correctness. Raw source fingerprints and original 3Sources/BBC notices identify the underlying public research releases and their conditions. No original raw articles are redistributed here. This archive is prepared locally; reviewer access requires attaching or hosting it through the permitted anonymous submission mechanism. No upload or public release is performed by this build script.

Original implementation files are included under `source/` for method inspection, with their original relative project layout. They cover source partitioning/features, the validated fit engine, the BBC adaptation, the full-density LP and the label-blind content/joint-bootstrap follow-up. They are byte-preserved source files, not an assertion that full training can run from this numeric-only archive: full training requires the published raw sources and the original project inputs. The standalone `recompute.py` is the executable offline reviewer entry.

Editable native LaTeX tables and their stylesheet are under editable_tables/. Black bold marks unrounded numerical minima within declared comparison groups, not significance. No delta columns or numeric outcome colors were added.

## Start here: core audit report

Read `report/REPORT.md` and `report/all_topic_ranges.csv`. They include all six native topics for each of the three declared sets, the scalar distance from each reference to its candidate envelope, and the unavoidable health shortfall for every convex mixture. Health is an exploratory illustration; topic outcomes are retained together.

Minimal numeric entry: install NumPy and threadpoolctl (`requirements-core.txt`), then run `python reproduce_core.py`. This checks the payload and recomputes Table 2, the four paired H intervals and coverage from frozen inputs. Original environment pins remain in `requirements.txt`. Optional plot entry: install `requirements-plot.txt`, then run `python reproduce_core.py --coverage-only --plot`. Generated reports go under `recomputed/`. Full training is not required for these core checks.

This is the review-ready local archive. Its reviewer delivery URL or permitted submission attachment is determined by the author; it has not been uploaded by these scripts.
