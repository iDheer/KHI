# Opinion Network Formation — Team KHI

DPCN Assignment 1. We build and analyse networks from a class survey of 60 Likert statements covering
**Technology, Education, Ethics & Society and Environment**.

**Final report:** [`Team_KHI_Assignment1_Report.pdf`](Team_KHI_Assignment1_Report.pdf) (LaTeX source in `report/`)

Team: **Inesh** (40%), **Khwaish** (30%), **Hiten** (30%). The detailed breakdown is in Section 7 of the report.

## Approach in one paragraph
Answers are coded from −2 to +2. "No Comments" and blank answers count as missing, and respondents who answered
fewer than 75% of the statements are removed (96 → 87). Two students are similar if their answers
**deviate from the class average in the same way**, measured as item-centred, pairwise-complete cosine similarity.
Each student is linked to their 8 most similar peers (symmetric kNN backbone), and Louvain finds the communities.
We check the result against a column-shuffling null model, degree-preserving rewiring, other graph constructions
and a resolution sweep. A second network links **statements** through FDR-significant Spearman correlations
and shows which opinions go together.

## Key findings
| | |
|---|---|
| Respondent network | 87 nodes, 466 edges, small-world (σ = 2.26) |
| Communities | **Moderate Pragmatists** (45), **Progressive Idealists** (23), **Tech-Cautious Regulators** (19); Q = 0.418 vs 0.363 ± 0.014 for shuffled data (z = 4.0) |
| Per-theme groupings | almost independent across themes (ARI ≤ 0.16) |
| Statement network | partly recovers the 4 themes (NMI = 0.47); a cross-theme *digital rights & regulation* cluster; 7 divisive statements (exams, attendance, online learning, AI diagnosis…) have no strong links |

## Repository layout
```
data/Survey_Results_UC.csv     raw survey export (96 rows × 61 columns)
docs/DPCN_Assignment_1.pdf     assignment brief
src/preprocess.py              loading, Likert encoding, quality filtering
src/network.py                 similarity measures, kNN / threshold graphs, statement correlation graph
src/analysis.py                metrics, Louvain + stability, null models, robustness, profiling
src/visualize.py               all figures
run_pipeline.py                end-to-end entry point
outputs/
  cleaned_likert_matrix.csv    87 × 60 numeric matrix used for the analysis
  metrics.json                 every number quoted in the report
  respondent_network.graphml   respondent network (open in Gephi / Cytoscape)
  statement_network.graphml    statement network
  figures/                     fig01 … fig10 (PNG)
  tables/                      quality report, k-sweep, centralities, community tests, robustness, …
report/report.tex              report source (pdflatex)
```

## Reproduce
```bash
pip install -r requirements.txt
python run_pipeline.py          # ~30 s; rewrites everything in outputs/
cd report && pdflatex report.tex && pdflatex report.tex
```
All random steps are seeded (Louvain seeds 0–99, permutation / rewiring seed 42), so a re-run gives the same results (differences only in the last floating-point digits).
Tested with Python 3.14, numpy 2.4, pandas 3.0, networkx 3.6, scipy 1.18, scikit-learn 1.9, matplotlib 3.11.
