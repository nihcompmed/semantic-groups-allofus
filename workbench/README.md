# Researcher Workbench code: the participant-level analysis

Aggarwal M, Periwal V. *Semantic Groups of Survey Items, Screening Thresholds, and Recorded Conditions in All of Us.*
JAMA Network Open (submitted).

These scripts run inside the All of Us Researcher Workbench, on the Registered Tier Curated Data Repository, version 9
(`R2025Q4R6`). Running them needs Registered Tier access. Participant-level data never leave the Workbench. Each step
that produces a result writes 1 small zip of aggregate files, and only that zip is downloaded. Before writing it, the
script checks every file against the All of Us Data and Statistics Dissemination Policy with `disclosure.py`: no count of
1 to 20, and no value from which such a count could be derived.

The semantic groups are built outside the Workbench, from the item text (`../local/`, README there). This folder holds
their result, the basis package in `basis/` (and `basis_tfidf/` for the word-count grouping), and uses it only to score
the answers. `items_151.csv` and `encoding_map_all151.csv` are the item table and the coding of every answer.

## Setup

Copy this folder into the Workbench (for example to `~/workbench_jno`) and run every command from it, in a terminal
(not a notebook cell):

```bash
cd ~/workbench_jno
python3 preflight_151.py
```

`preflight_151.py` checks the folder (the item table, the basis package, the embeddings) and must print
`preflight clean.` before step 1.

Every script reads its settings from `wb_config.py`: the item table (`items_151.csv`), the basis package (`basis/`) and
the response folder `responses_out_items_151/`, which step 1 writes beside the scripts. `WORKSPACE_CDR` is set by the
Workbench. Python is the Workbench's own `python3`; steps 10 and 14 also use a separate environment for PheTK:

```bash
python3 -m venv ~/phetk_env
~/phetk_env/bin/pip install phetk==0.3.6 google-cloud-bigquery-storage
```

(PheTK 0.3.6 needs numpy below 2.0, hence its own environment.)

## Steps, in order

| Step | Command | Writes (stays on the Workbench) | Download | In the paper |
|---|---|---|---|---|
| 1 | `python3 build_responses_v9.py` | `responses_out_items_151/`: the cohort's answers, imputed (`responses_k5.csv`), the imputation mask, age and sex | the counts in `run_config.json`, `pull_summary.json`, `cohort_sizes.csv` | Methods, Participants; eMethods 2 |
| 2 | `python3 k_choice.py` | | `k_choice_<CDR>.zip` | eFigure 1, panels C to E |
| 3 | `python3 screen_battery.py` | `screen_out/scores_<encoder>.csv`: each participant's distance under each of the 6 encoders | | Methods, Distance; eMethods 2 |
| 4 | `python3 screen_tfidf.py`, `python3 mca_basis.py --axes 100 --parallel 10`, `python3 comparators.py`, `python3 comparators.py --mca-k horn` | `screen_out/scores_<grouping>.csv` for the 6 other groupings | | eMethods 5; steps 11 to 14 run every grouping |
| 5 | `python3 cutoffs.py` | `screen_out/cutoff_flags.csv`: each participant's 6 screening thresholds | | eMethods 3 |
| 6 | `python3 zip_cutoffs.py` | | `cutoffs_<CDR>.zip` | eTable 5 |
| 7 | `python3 care.py --outcomes-only` | `screen_out/care_outcomes_per_person.csv`: barriers to care and the emergency department share | | eMethods 3 |
| 8 | `python3 explore_other_data.py --cells 9`, then `--cells 10` | `sleep_person_local.csv`, `wear_enrollees_local.csv`, `answer_dates_local.csv` | | eMethods 3 |
| 9 | `python3 sleep_reliability.py` | | `sleep_reliability_<CDR>.zip` | eMethods 3, the 14-night rule |
| 10 | `~/phetk_env/bin/python phecode_coverage.py` | `screen_out/phetk/`: each participant's diagnosis codes and phecode counts | `phecode_coverage_<CDR>.zip` | eMethods 3 |
| 11 | `python3 outcome_deciles.py` | | `outcome_deciles_<CDR>.zip` | Figure 2A to C, eTable 7; eFigure 4 and eTable 13 (all 12 groupings) |
| 12 | `python3 ed_check.py` | `screen_out/ed_check/visits_person_local.csv` | `ed_check_<CDR>.zip` | Figure 2B (emergency department), eMethods 3 |
| 13 | `python3 sleep_device_deciles.py` | | `sleep_device_deciles_<CDR>.zip` | eTable 6 |
| 14 | `~/phetk_env/bin/python phecode_counts_jno.py`, then `python3 conditions_jno.py`, then `python3 conditions_deciles_jno.py` | `screen_out/phetk/phecode_counts_cohort_jno_local.tsv`, `screen_out/test_checks_jno_local.csv` | `phecode_counts_jno_<CDR>.zip`, `condition_tests_jno_<CDR>.zip`, `conditions_deciles_jno_<CDR>.zip` | Figure 2D, eFigure 2, eTable 7; eMethods 3; Supplement 2 (test codes) |
| 15 | `python3 covariates.py` | `screen_out/covariates_per_person.csv` | `table1_<CDR>.zip` | eTables 2 and 3 |
| 16 | `python3 env_versions.py` | | `env_versions_<CDR>.zip` | the Python and library versions |
| 17 | `python3 direction_own.py`, then `python3 direction_own.py --invisible` | | `direction_own_<CDR>.zip`, `direction_own_invisible_<CDR>.zip` | Figure 3, eTable 8; eMethods 2 |
| 18 | `python3 conditions_assoc_jno.py --encoder <encoder>`, then `python3 conditions_assoc_adjusted_jno.py --encoder <encoder>`, for each of the 6 encoders; then `python3 breast_timing_jno.py` | `screen_out/survey_anchor_local.csv` (each participant's survey date) | `conditions_assoc_jno_<encoder>_<CDR>.zip`, `conditions_assoc_adjusted_jno_<encoder>_<CDR>.zip`, `breast_timing_jno_<CDR>.zip` | Figure 4, eFigure 3, eTables 9 to 11; eMethods 4 |

Each download is printed as `download -> screen_out/<name>_<CDR>.zip`. After step 18, `python3 collect_conditions_jno.py`
gathers the zips of steps 14 and 18 into `~/conditions_jno_<CDR>.zip`.

Step 17 needs steps 3 and 5; step 18 needs step 14. Steps 1, 8 and 10 read the repository's BigQuery tables and take
minutes each (step 8's answer dates are a 7.8 GB scan); the others take seconds to a few minutes.

`env_versions.py` writes the Python and library versions on their own. `care_answer_check.py` (below) records them in
its summary file under `versions`, and the versions in the Methods are the ones it recorded.

## Check of the barrier answers

| Command | Download | What it checks |
|---|---|---|
| `python3 care_answer_check.py` (after step 7) | `care_answer_check_<CDR>.zip` | how `care.py` reads the stored answers to the 21 barrier questions, and how many participants gave no usable answer (eMethods 3) |

## After the Workbench

Each downloaded zip unpacks into `../local/outputs/workbench/R2025Q4R6/<name>/`, where the copies used for the paper
already are. The tables and figures are drawn there, never on the Workbench, by the run scripts in `../local/scripts/`
(`../local/README.md`).

## Other files in this folder

- Imported by the steps: `disclosure.py` (the dissemination policy), `wb_config.py` (the settings),
  `zip_screen_aggregates.py` (the check of every download), and `breadth_sweep.py`, `breadth_exclusive.py`,
  `care_sweep.py`, `care_adjusted.py`, `sleep_sweep.py`, `sleep_adjusted.py`, `phecode_sweep.py`, `phecode_deciles.py`,
  `phecode_deciles_pooled.py`, `phecode_directions.py`, `phecode_block.py`, `phecode_cross.py`, `phecode_fifteen.py`,
  `phecode_fifteen_adjusted.py` and `phecode_screened.py`, whose functions the steps reuse.
- Not used for the results in the paper: `agreement_curves.py`, `care_band_check.py`, `care_covariates.py`,
  `care_exclusive.py`, `care_records.py`, `ceiling.py`, `construct_labels.py`, `exemplar_encoder.py`, `fig_directions.py`,
  `matched_breadth.py`, `phecode_event_dates.py`, `phecode_screen_negative.py`, `phecode_screen_negative_directions.py`,
  `phecode_screen_negative_timing.py`, `semantic_groups.py` and `sleep_by_model.py`. They stay because
  `preflight_151.py` checks that several of them are present.
