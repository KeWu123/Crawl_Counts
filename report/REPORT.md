# Counting-target audit: core results

Recomputed from frozen per-record laws, topic labels and the supplied story reference. No fitting or threshold search is performed.

## Candidate coverage

| Candidate set | R (%) | Center-to-reference TV (%) | Topics outside / 6 | Minimum health shortfall (pp) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.898734 | 13.024610 | 5 | 2.854187 |
| content050 | 8.954271 | 12.998952 | 4 | 2.854187 |
| content070 | 2.339693 | 12.865237 | 4 | 2.677082 |

For a topic indicator f, E_mix[f] is the convex combination of candidate means. Its attainable scalar interval is [min mean, max mean]. The reference health share is above every candidate in each tested set, so every allowed mixture underestimates it. This concerns these declared finite candidate sets.

Health is an exploratory illustration from the baseline audit; all six native topics are reported.

## All native topics

| Set | Topic | Candidate min (%) | Candidate max (%) | Story reference (%) | Distance (pp) | Direction |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | business | 27.320675 | 27.505504 | 25.000000 | 2.320675 | overestimate |
| baseline | entertainment | 14.419609 | 14.451477 | 14.423077 | 0.000000 | inside |
| baseline | health | 9.971595 | 10.126582 | 12.980769 | 2.854187 | underestimate |
| baseline | politics | 11.568175 | 11.708861 | 11.778846 | 0.069985 | underestimate |
| baseline | sport | 24.050633 | 24.108862 | 21.394231 | 2.656402 | overestimate |
| baseline | tech | 12.341772 | 12.426254 | 14.423077 | 1.996823 | underestimate |
| content050 | business | 27.320675 | 30.450450 | 25.000000 | 2.320675 | overestimate |
| content050 | entertainment | 14.054054 | 14.451477 | 14.423077 | 0.000000 | inside |
| content050 | health | 9.971595 | 10.126582 | 12.980769 | 2.854187 | underestimate |
| content050 | politics | 10.945946 | 11.708861 | 11.778846 | 0.069985 | underestimate |
| content050 | sport | 21.081081 | 24.108862 | 21.394231 | 0.000000 | inside |
| content050 | tech | 12.341772 | 13.468468 | 14.423077 | 0.954608 | underestimate |
| content070 | business | 27.320675 | 27.657267 | 25.000000 | 2.320675 | overestimate |
| content070 | entertainment | 14.419609 | 14.533623 | 14.423077 | 0.000000 | inside |
| content070 | health | 9.971595 | 10.303688 | 12.980769 | 2.677082 | underestimate |
| content070 | politics | 11.568175 | 11.822126 | 11.778846 | 0.000000 | inside |
| content070 | sport | 23.535792 | 24.108862 | 21.394231 | 2.141561 | overestimate |
| content070 | tech | 12.147505 | 12.426254 | 14.423077 | 1.996823 | underestimate |

For each topic separately, mixture expectations fill the interval between candidate means. A positive distance is unavoidable for every mixture in that declared set; zero does not establish a jointly attainable six-topic vector.

## Distinct diagnostics

Linkage precision/recall compares partition membership. TV compares per-record counting weights, which depend on K and the size m(i) of each record's group. Two different pairings of four records into two size-two groups have identical 1/4 record weights and zero counting TV even with disjoint linked pairs.

Exact candidate ranges, center-plus/minus-radius bands, and paired bootstrap intervals are different quantities. The cosine .50 center radius band covers health; its exact candidate range still excludes the reference.

## Reproduce

Run `python reproduce_core.py` from the extracted artifact. It verifies payload checksums, recomputes all 77 Table 2 cells and all four H intervals (5,000 joint test-cluster resamples each), plus the original controls and extension results, then writes this diagnostic under `recomputed/`.

Optional: `python reproduce_core.py --coverage-only --plot` produces a standalone PNG/SVG rendering of the Figure 2(a,b) data. It uses the frozen arrays; the manuscript figure remains the selected original.

H is a post-hoc conditional description with models, partitions and splits fixed. All six topics are reported without selecting outcomes. Raw article text is not redistributed.
