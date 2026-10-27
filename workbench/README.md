# Researcher Workbench code: the participant-level analysis

These scripts run inside the All of Us Researcher Workbench on the Registered Tier Curated Data Repository, version 9
(`R2025Q4R6`); running them needs Registered Tier access. Participant-level data never leave the Workbench. Each step
that produces a result writes a small zip of aggregate files to `downloads/`, and only those zips are downloaded. Before a
file is zipped, `release.py` applies the All of Us Data and Statistics Dissemination Policy: no participant count of 1 to
20, and no value from which such a count could be derived.

The semantic groups are built outside the Workbench, from the item text (`../local/`). This folder holds their result,
the basis packages in `basis/` (the 6 sentence encoders) and `basis_tfidf/` (the 2 word-count groupings), and uses them
only to score the answers.

## Setup

Copy this folder into the Workbench, for example to `~/semantic_groups`, and run every command from it in a terminal:

```bash
cd ~/semantic_groups
python3 preflight.py
```

`preflight.py` checks the folder and the environment and must print `preflight clean.` Python is the Workbench's own
`python3`. Step 4 uses PheTK 0.3.6, which needs numpy below 2.0, so it runs in its own environment:

```bash
python3 -m venv ~/phetk_env
~/phetk_env/bin/pip install phetk==0.3.6 google-cloud-bigquery-storage
```

## Steps

Run them in order, either one by one or all at once with `bash run_workbench.sh` (a log per step in `logs/`; `bash
run_workbench.sh 7` starts at step 7). Steps 1, 4 and 5 query the data repository; a rerun reuses what they pulled
(`--refresh` pulls again).

| Step | Command | What it computes | Download | In the paper |
|---|---|---|---|---|
| 1 | `python3 step1_responses.py` | the cohort and its answers to the 151 items, with up to 5 missing answers imputed | `responses_counts` | Materials and methods, Study population and data; Appendix S1 |
| 2 | `python3 step2_scores.py` | every participant's distance under the 12 groupings and under the factor model at k = 3; Horn's k of PCA, the factor model and MCA | `k_choice` | Figure S1; Appendix S4 |
| 3 | `python3 step3_thresholds.py` | the six screening thresholds per participant | `thresholds` | Table S3 |
| 4 | `~/phetk_env/bin/python step4_phecodes.py` | each participant's diagnosis codes counted by phecode | | the recorded conditions |
| 5 | `python3 step5_outcomes.py` | barriers to care, the emergency department share, Fitbit sleep, the tests behind 3 conditions' populations, the visits | `sleep_reliability`, `care_answers` | Materials and methods, Outcomes; Appendix S3 |
| 6 | `python3 step6_cohort.py` | the cohort table, and the participants against the rest of the release | `cohort_table`, `selection` | Supplementary Files 2 and 3 |
| 7 | `python3 step7_deciles.py` | every outcome by decile of the distance, adjusted for age and sex, in the cohort and below every threshold | `deciles`, `deciles_ed`, `ed_visits` | Figures 2 and 5; Appendix S3 |
| 8 | `python3 step8_contributors.py` | the semantic groups that carry each outlier's distance | `contributors` | Figure 3 |
| 9 | `python3 step9_factor_comparison.py` | the outliers and most atypical tenth under each encoder against those under the factor model | `factor_decile10`, `factor_outliers`, `factor_below`, `factor_outliers_ed`, `factor_sources` | Figures 4 and 6; Appendix S4 |
| 10 | `python3 step10_conditions.py` | the semantic groups (and the factors) and the 8 recorded conditions | `conditions`, `conditions_adjusted`, `factor_conditions` | Figure 7; Appendix S5 |

Each script's docstring states what it computes, what it reads and what it writes.

## Folders

- `responses/`: the cohort and its answers (step 1).
- `work/`: the person-level files the steps pass to each other. They never leave the Workbench.
- `downloads/`: the aggregate files of each step, and `<name>_R2025Q4R6.zip` to download.

## Modules the steps share

| File | What it holds |
|---|---|
| `settings.py` | the folders, the 6 encoders, the fixed settings |
| `release.py` | the dissemination policy, the checks every file passes before it is zipped, the zip |
| `scoring.py` | the answers, the scoring of a grouping and the distance; PCA, the factor model, MCA; the core of a group |
| `outcomes.py` | the per-person outcomes read back from `work/`, the 8 recorded conditions, the statistics of one set |
| `models.py` | the joint model of a category with age and sex, the contrast between 2 sets, the trend, the logistic model |
| `deciles.py` | the deciles, the pooling of sparse deciles, the hiding across deciles, the decile tables |
| `comparison.py` | the factor models and their rotations, how a distance divides among items and sources, the unit counts |

The data files are `items_151.csv` (the items: order, scale, keying, instrument, construct), `encoding_map_all151.csv`
(every answer of every item and its numeric value) and `standalone_sources.csv` (the study each stand-alone question was
taken from).

## After the Workbench

Unzip each download into `../local/outputs/workbench/R2025Q4R6/`, where the copies used for the paper already are (each
zip unpacks into a folder of its own name). The figures and tables are drawn there, never on the Workbench, by
`../local/run_paper.sh` (`../local/README.md`).
