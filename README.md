# Semantic groups of survey items in All of Us

Code for: Aggarwal M, Periwal V. *Semantic language models detect distress more precisely than factor models.*

The paper groups 151 mental health and social survey items of the All of Us Research Program by meaning, from the item
text alone, with sentence encoders (semantic groups). It scores every participant's answers on the groups, measures
how far each participant's scores lie from the cohort's mean scores, and compares the most atypical participants with
those under a factor model fitted to the same answers. The outcomes are screening thresholds met, recorded conditions
in the electronic health record, barriers to care, and sleep recorded by wearable devices.

## Contents

- `workbench/` runs on the All of Us Researcher Workbench, on participant-level data, in 10 steps. Each step that
  produces a result writes a small zip of aggregate files to download. Start with `workbench/README.md`.
- `local/` runs outside the Workbench, on a CPU. `local/semantic_groups/` builds the semantic groups from the item text
  for 6 sentence encoders, and `local/paper/` draws every figure, appendix table and supplementary file of the paper
  from the downloads, which ship in `local/outputs/workbench/R2025Q4R6/`. Start with `local/README.md`.

## Where each result comes from

Each display item has its own script in `local/paper/`, named after it. The downloads are in
`local/outputs/workbench/R2025Q4R6/<download>/`, one folder per download, named as in `workbench/README.md`. Results
marked "semantic groups" are built locally from the item text (`local/run_semantic_groups.sh`) and need no Workbench
data.

| In the paper | Drawn by (`local/paper/`) | Reads the download(s) | Workbench step |
|---|---|---|---|
| Figure 1 | `figure1.py` | (semantic groups) | |
| Figure 2 and its figure supplements 1 and 2 | `figure2.py`, `figure2_supplement1.py`, `figure2_supplement2.py` | `deciles`, `deciles_ed` | 7 |
| Figure 3 | `figure3.py` | `contributors` | 8 |
| Figure 4 and its figure supplements 1 and 2 | `figure4.py` | `factor_outliers`, `factor_outliers_ed`, `factor_sources`, `factor_decile10` | 9 |
| Figure 5 and its figure supplement 1 | `figure5.py`, `figure5_supplement1.py` | `deciles`, `deciles_ed` | 7 |
| Figure 6 and its figure supplements 1 and 2 | `figure6.py` | `factor_below`, `factor_sources`, `factor_decile10` | 9 |
| Figure 7 and its figure supplements 1 to 5 | `figure7.py` | `conditions` | 10 |
| Figures 2 to 7, source data 1 | `figure2_source_data.py` to `figure7_source_data.py` | as the figure, and `conditions_adjusted` for Figure 7 | as the figure |
| Appendix 2–figure 1 | `appendix2_figure1.py` | `k_choice` (and the semantic groups) | 2 |
| Appendix 2–figure 2 | `appendix2_figure2.py` | (semantic groups) | |
| Appendix 2–tables 1 and 2 | `appendix2_table1.py`, `appendix2_table2.py` | (semantic groups) | |
| Appendix 3–table 1 | `appendix3_table1.py` | `thresholds` | 3 |
| Appendix 4–table 1 | `appendix4_table1.py` | `factor_decile10` | 9 |
| Appendix 5–table 1 | `appendix5_table1.py` | `conditions`, `factor_conditions` | 10 |
| Supplementary file 1 | `supplementary_file1.py` | (semantic groups and the item table), `factor_decile10` | 9 |
| Supplementary file 2 | `supplementary_file2.py` | `cohort_table` | 6 |
| Supplementary file 3 | `supplementary_file3.py` | `selection` | 6 |
| Numbers given only in the text: the 14-night rule for sleep, the emergency department visits | (quoted in the text) | `sleep_reliability`, `ed_visits` | 5, 7 |

To follow one result: open its script in `local/paper/`, whose docstring names the files it reads; find the download
in the table of steps in `workbench/README.md`; open that step's script. `local/run_paper.sh` runs every script of
`local/paper/` and compares the output with `local/expected/`.

## Data

This repository holds no participant-level data. The participant data are from the All of Us Research Program,
Registered Tier, Curated Data Repository version 9. Registered researchers can access them on the All of Us Researcher
Workbench (https://www.researchallofus.org). The files in `local/outputs/workbench/` hold aggregate statistics only,
checked against the All of Us Data and Statistics Dissemination Policy before they left the Workbench.

## Licenses

The code is released under the MIT License (`LICENSE`). The repository includes item wording only for the PHQ-9, GAD-7,
and ACE items and the stand-alone All of Us questions. The All of Us Survey Codebooks give the wording of every other
item under its item code (`local/README.md`, "Item wording"). The license does not cover the item wording, which belongs
to the owners of each questionnaire and to the All of Us Research Program. It also does not cover the sentence-encoder
models or the SUBTLEX-US word frequencies, which are downloaded separately under their own terms (`local/README.md`).

## Contact

Manu Aggarwal, National Institutes of Health (manu.aggarwal@nih.gov).
