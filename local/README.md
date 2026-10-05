# Local code

Aggarwal M, Periwal V. *Semantic Groups of Survey Items, Screening Thresholds, and Recorded Conditions in All of Us.*
JAMA Network Open (submitted).

This folder holds everything that runs outside the All of Us Researcher Workbench:

- **The semantic groups.** They are built from the text of the 151 survey items, for 6 sentence encoders, with
  gte-large-en-v1.5 specified in advance as the main one. This part uses the item text and the stored item embeddings
  only, never a participant's answers.
- **The tables and figures of the results.** They are drawn from aggregate files written on the Researcher Workbench,
  which ship in `outputs/workbench/R2025Q4R6/`. The code that writes them is in `../workbench/` (its `README.md` gives
  the steps in order).

Everything here runs on a CPU. No participant-level data are in this folder.

## Where each table and figure comes from

| In the paper | Run | Script |
|---|---|---|
| Figure 1 | `run_semantic_groups.sh` | `fig_method_jno.py` |
| Figure 2, eFigure 2, eTables 5 to 7 | `run_deciles.sh` | `fig_outcome_deciles.py`, `fig_conditions_deciles.py`, `etable_thresholds_jno.py`, `etable_sleep_device_jno.py`, `etable_deciles_jno.py` |
| Figure 3, eTable 8 | `run_contributions.sh` | `fig_content_bars_jno.py`, `etable_content_jno.py` |
| Figure 4, eFigure 3, eTables 9 to 11 | `run_conditions.sh` | `fig_conditions_assoc.py`, `etable_conditions_jno.py` |
| eFigure 1 | `run_semantic_groups.sh` | `fig_variance_explained.py` |
| eFigure 4, eTable 13 | `run_groupings.sh` | `fig_groupings_deciles.py`, `etable_groupings_jno.py` |
| eTable 1 | `run_semantic_groups.sh` | `etable1_items.py` |
| eTables 2 and 3 | `run_cohort.sh` | `table1_jno.py` |
| eTable 4 | `run_semantic_groups.sh` | `etable_groups_jno.py` |
| eTable 12 | `run_semantic_groups.sh` | `etable_switch_items.py` |
| Supplement 2 | `run_semantic_groups.sh`, and `supp_condition_tests_jno.py` on its own | `supp_items_jno.py`, `etable_switch_items.py`, `supp_condition_tests_jno.py` |

Numbers quoted in the text but not drawn come from the shipped Workbench files directly: the breast cancer timing in the
Results (`breast_timing_jno/`), the 14-night rule for sleep (`sleep_reliability/`), and the counts in eMethods 3
(`ed_check/`, `condition_tests_jno/`, `phecode_counts_jno/`).

## Layout

```
README.md         this file
ross_proj/        the fitting library (parallel analysis and the sparse fit)
scripts/          every script; the run_*.sh scripts run them in order
data/             the inputs of the semantic groups ("Inputs" below); data/items_151.csv is rebuilt by step 2
outputs/          written by the runs, except the inputs saved from other steps: outputs/workbench/R2025Q4R6/ (the
                  Researcher Workbench files) and outputs/k_selection/tfidf_subtlex/ (eFigure 1, panel B)
expected/         the files of record, compared with each run by check_expected.py
```

## Environment

Python 3.11 with these library versions:

```bash
python3.11 -m venv env
env/bin/pip install numpy==2.4.6 scipy==1.17.1 pandas==2.3.3 scikit-learn==1.9.0 matplotlib==3.11.0 \
    jax==0.10.1 jaxlib==0.10.1 optax==0.2.8 tqdm==4.68.2
```

The scripts set `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu` themselves. Every run below is started from this
folder as `PY=env/bin/python bash scripts/<run>.sh` and ends with `check_expected.py`, which compares the result with
`expected/` and prints `N of N match the files of record` when they are the same.

## The semantic groups

```bash
PY=env/bin/python bash scripts/run_semantic_groups.sh
```

The run takes about 30 to 40 minutes on a 48-core CPU: step 3 about 7 minutes, step 5 about 20 minutes (600 fits of
about 10 seconds each, 6 jobs in parallel). It ends with `check_expected.py semantic_groups` (14 files).
`random_starts.py` skips any start whose files already exist; to repeat step 5 from scratch, delete
`outputs/random_starts/` first.

| Step | Script | Reads | Writes | In the paper |
|---|---|---|---|---|
| 1 | `build_text_of_record.py` | `data/sources/administered_text_v9.csv`, `core_items_151.csv` | `data/sources/items_text_of_record.csv` | the text embedded (eMethods 1) |
| 2 | `build_items_table.py` | `data/sources/*`, `data/encoding_map_all151.csv` | `data/items_151.csv` | the item table |
| (0) | `0_embed.py` | `data/items_151.csv`, the models | `data/embeddings/admin151_items_<encoder>.npz` | optional ("Encoding the item text again") |
| 3 | `1_select_k.py` | the embeddings | `outputs/k_selection/` | the number of groups, by Horn's parallel analysis |
| 4 | `2_fit.py` | the embeddings, k | `outputs/fits/` | the fit started from the principal components |
| 5 | `random_starts.py` | the embeddings, k | `outputs/random_starts/<encoder>__k<k>_ls1.0/` | 100 random orthonormal starts per encoder |
| 6 | `select_projection.py` | steps 4 and 5 | `outputs/random_starts/selection.json`, `candidates.csv` | the selection rule (eMethods 1) |
| 7 | `materialize_selection.py` | `selection.json` | `outputs/basis_fit/<encoder>/` | the basis of each encoder |
| 8 | `cores.py` | the basis, `data/items_151.csv` | `outputs/cores/` | the core items of every group (eMethods 1) |
| 9 | `build_basis_package.py` | the basis, the cores | `outputs/basis/` | the package the Workbench code reads (a copy is in `../workbench/basis/`) |
| 10 | `xenc.py` | the cores | `outputs/xenc/` | agreement across encoders (eMethods 1) |
| 11 | `etable1_items.py` | `data/items_151.csv` | `outputs/tables/etable1_items.tex` | eTable 1 |
| 11 | `etable_groups_jno.py` | `outputs/basis/components.csv` | `outputs/tables/etable_groups_jno.{tex,csv}` | eTable 4 |
| 11 | `etable_switch_items.py` | `components.csv`, `data/items_151.csv` | `outputs/tables/etable_switch_items.tex`, `switch_items.csv` | eTable 12 and its Supplement 2 file |
| 11 | `supp_items_jno.py` | `data/items_151.csv`, `data/encoding_map_all151.csv`, `components.csv` | `outputs/supplementary_data_jno/` | Supplement 2 (item text, answer coding, group cores) |
| 11 | `fig_method_jno.py` | `data/items_151.csv`, `components.csv` | `outputs/method/fig_method_jno.{pdf,png}` | Figure 1 |
| 11 | `fig_variance_explained.py` | `outputs/k_selection/`, the 2 saved inputs | `outputs/k_selection/fig_variance_explained.{pdf,png}` | eFigure 1 |
| 12 | `check_expected.py semantic_groups` | `outputs/`, `expected/` | (prints) | the check |

`_common.py` holds the paths, the encoder list and the embedding loader. `construct_labels.py` holds the construct names
printed in the tables and figures.

### Inputs

In `data/`:

- `sources/administered_text_v9.csv`: the question string of each item as administered, from the survey metadata of
  the All of Us Curated Data Repository, version 9. It holds no participant data.
- `sources/core_items_151.csv`: each item's concept ID, survey and published wording.
- `sources/construct_map_151.csv`: each item's questionnaire and documented construct.
- `sources/scales_prism.csv`: each item's response range and reverse coding.
- `encoding_map_all151.csv`: the numeric value of every answer. The Workbench code uses it too, and step 2 checks the
  response ranges against it.
- `embeddings/admin151_items_<encoder>.npz`: the item embeddings. Each holds 151 unit-length vectors of 768
  (all-mpnet-base-v2) or 1024 numbers, computed by `0_embed.py` from `text_of_record`.
- `embeddings/admin151_items_tfidf-subtlex-k4.npz` and `..._k25.npz`: the word-count (TF-IDF) item vectors, 151 x 586,
  the same vectors under 2 names ("The other groupings").

In `outputs/`, saved from other steps and read only by `fig_variance_explained.py` (eFigure 1):

- `workbench/R2025Q4R6/k_choice/`: panels C to E, the eigenvalues and inertia of the groupings estimated from the
  answers, written on the Workbench by `../workbench/k_choice.py`.
- `k_selection/tfidf_subtlex/parallel_analysis.json`: panel B, the parallel analysis of the word-count grouping, written
  by `tfidf_horn.py` (below).

### Item wording

The wording of an item is included only where its questionnaire's terms permit redistribution: the PHQ-9, GAD-7, and
ACE items and the 13 stand-alone All of Us questions (eMethods 1 of the paper). For the other 112 items, the text
columns of `data/sources/*.csv`, `expected/data/items_151.csv`, `expected/outputs/tables/switch_items.csv`, and
`../workbench/items_151.csv` are empty. The All of Us Survey Codebooks
(https://support.researchallofus.org/hc/en-us/articles/360051991531) give the wording of every item under its item
code, the `item` column.

Every step of this README runs without that wording, because every step after the item table reads the stored
embeddings. Steps 1 and 2 rebuild the item table as shipped, with the same empty columns. Two optional checks need the
full wording, entered in `data/sources/administered_text_v9.csv` and `data/sources/core_items_151.csv`: encoding the
item text again (below) and rebuilding the word-count vectors with `build_tfidf_embeddings.py` ("The other groupings").

The word-count scripts (`tfidf_ref.py`, `tfidf_horn.py`, `build_tfidf_embeddings.py`) need the SUBTLEX-US word
frequencies (Brysbaert and New 2009), which are not redistributed here. Download
https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip and put the file
`SUBTLEXus74286wordstextversion.txt` (md5 `ee7be8db21f26e1925ac0ea3ab598f32`) in `data/subtlexus/`, or pass its path
with `--subtlex`.

### Encoding the item text again (optional)

The run starts from the stored embeddings. Encoding the text again is a check, not a step of the run, and it needs the
full item wording ("Item wording" above). On a CPU, 4
encoders reproduce the stored vectors bit for bit, gte-large-en-v1.5 at a cosine similarity of 1.000000 and
Qwen3-Embedding-0.6B at 0.9998 or more. A fit on slightly different vectors can differ slightly, which is why the run
uses the stored ones.

The models, from the Hugging Face Hub, at these revisions:

| Encoder | Hub name | Revision |
|---|---|---|
| bge-m3 | BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 |
| bge-large-en-v1.5 | BAAI/bge-large-en-v1.5 | d4aa6901d3a41ba39fb536a557fa166f842b0e09 |
| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 |
| e5-large-v2 | intfloat/e5-large-v2 | f169b11e22de13617baa190a028a32f3493550b6 |
| gte-large-en-v1.5 | Alibaba-NLP/gte-large-en-v1.5 | 104333d6af6f97649377c2afbde10a7704870c7b |
| qwen3-embedding-0.6b | Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |

Two environments, both with torch 2.12.0 (CPU):

- 5 encoders: sentence-transformers 5.5.1 and transformers 5.10.2.
- gte-large-en-v1.5: sentence-transformers 3.4.1 and transformers 4.49.0, because its published custom code
  (Alibaba-NLP/new-impl, revision 40ced75c3017eb27626c9d4ea981bde21a2662f4) does not run under transformers 5.

Download each model into a folder named after the encoder (for example `models/bge-m3/`), then, from `scripts/`:

```bash
CUDA_VISIBLE_DEVICES="" ENV5/bin/python 0_embed.py --model bge-m3 --models-dir ../models --out /tmp/bge-m3.npz
CUDA_VISIBLE_DEVICES="" ENV4/bin/python 0_embed.py --model gte-large-en-v1.5 --models-dir ../models --trust-remote-code --out /tmp/gte.npz
```

Without `--out`, `0_embed.py` overwrites the stored embeddings in `data/embeddings/`.

### Expected results

| Encoder | k | Retained start | Groups on 1 construct | Median core size | Items not in any core |
|---|---|---|---|---|---|
| bge-m3 | 27 | seed 41 | 11 | 4 | 27 |
| bge-large-en-v1.5 | 26 | seed 74 | 13 | 4 | 31 |
| all-mpnet-base-v2 | 23 | seed 28 | 8 | 5 | 32 |
| e5-large-v2 | 26 | seed 64 | 15 | 4 | 34 |
| gte-large-en-v1.5 | 27 | seed 95 | 13 | 4 | 28 |
| qwen3-embedding-0.6b | 25 | seed 32 | 12 | 4 | 30 |

Across encoders, the median Jaccard index of matched cores is 0.67, against 0.14 with item labels shuffled
(`outputs/xenc/summary.json`).

### Device dependence

The sparse fit runs 5000 steps of a nonconvex optimization. On a GPU, or on a CPU with other library versions, the
arithmetic can round differently, and the fit can come out different in the last digits of W. The groups reported in
the paper are the core items in `outputs/basis/components.csv`, so `check_expected.py` compares that file on its own.
The Workbench code reads the copy of the basis package in `../workbench/basis/`.

## Cohort (eTables 2 and 3)

`../workbench/covariates.py` (Workbench step 15) writes the cohort's characteristics as aggregate counts. They ship in
`outputs/workbench/R2025Q4R6/table1/table1_covariates.csv`, without the rows for race and ethnicity, which this study
did not use.

```bash
PY=env/bin/python bash scripts/run_cohort.sh
```

| Script | Writes | In the paper |
|---|---|---|
| `table1_jno.py` | `outputs/tables/etable_encoders_jno.{tex,csv}`, `etable_subsets_jno.{tex,csv}`, and `table1_jno.{tex,csv}` (a 2-column version, not in the paper) | eTables 2 and 3 |
| `check_expected.py cohort` | (prints) | the check (4 files) |

## Outcomes by decile of the distance (Figure 2, eFigure 2, eTables 5 to 7)

These files, written on the Workbench (`../workbench/README.md`, steps 1 to 14), ship in `outputs/workbench/R2025Q4R6/`:

| Folder | Workbench script | What it holds |
|---|---|---|
| `outcome_deciles/` | `outcome_deciles.py` | thresholds met, barriers to care and sleep by decile, under all 12 groupings, adjusted for age and sex |
| `ed_check/` | `ed_check.py` | the emergency department share by decile among participants with coded care, and the visit counts of eMethods 3 |
| `conditions_deciles_jno/` | `conditions_deciles_jno.py` | the 8 recorded conditions by decile, 6 encoders, each in its own population |
| `condition_tests_jno/` | `conditions_jno.py` | the codes that mark a participant as tested (Supplement 2) and the population of each condition (eMethods 3) |
| `phecode_counts_jno/` | `phecode_counts_jno.py` | the case counts with and without the 3 removed codes (eMethods 3) |
| `sleep_device_deciles/` | `sleep_device_deciles.py` | sleep by decile within each device source |
| `cutoffs/` | `cutoffs.py`, `zip_cutoffs.py` | the 6 screening thresholds, evaluable and met |
| `sleep_reliability/` | `sleep_reliability.py` | the 14-night rule of eMethods 3 (quoted in the text, not drawn) |

```bash
PY=env/bin/python bash scripts/run_deciles.sh
```

| Script | Writes | In the paper |
|---|---|---|
| `fig_outcome_deciles.py` | `outputs/outcome_deciles/fig_r2_deciles.{pdf,png}` | Figure 2 |
| `fig_conditions_deciles.py` | `outputs/outcome_deciles/fig_conditions_deciles.{pdf,png}` | eFigure 2 |
| `etable_thresholds_jno.py` | `outputs/tables/etable_thresholds_jno.{tex,csv}` | eTable 5 |
| `etable_sleep_device_jno.py` | `outputs/tables/etable_sleep_device_jno.{tex,csv}` | eTable 6 |
| `etable_deciles_jno.py` | `outputs/tables/etable_deciles_jno.{tex,csv}` | eTable 7 |
| `check_expected.py deciles` | (prints) | the check (6 files) |

`conditions_local.py` holds the 8 recorded conditions (keys, labels, populations) for every script that draws them.
`supp_condition_tests_jno.py`, run on its own, writes the Supplement 2 file of test codes
(`outputs/supplementary_data_jno/condition_test_codes.csv`) from `condition_tests_jno/`.

## Groups contributing most to the distance (Figure 3, eTable 8)

`../workbench/direction_own.py` (Workbench step 17) divides each outlier's squared distance into 1 part per group and
counts how often each group is among a participant's 3 largest parts, for all outliers and, with `--invisible`, for the
outliers who could be evaluated on all 6 screening thresholds and did not meet any (eMethods 2). Its files ship in
`outputs/workbench/R2025Q4R6/direction_own/` (`direction_top3_own.csv` and `direction_top3_own_invisible.csv`; the 2
`direction_summary_*` files are not used in the paper).

```bash
PY=env/bin/python bash scripts/run_contributions.sh
```

| Script | Writes | In the paper |
|---|---|---|
| `fig_content_bars_jno.py` | `outputs/direction_own/fig_content_bars_jno.{pdf,png}` | Figure 3 |
| `etable_content_jno.py` | `outputs/tables/etable_content_jno.{tex,csv}` | eTable 8 |
| `check_expected.py contributions` | (prints) | the check (2 files) |

## Semantic groups and recorded conditions (Figure 4, eFigure 3, eTables 9 to 11)

On the Workbench (step 18), `conditions_assoc_jno.py` fits, for each encoder and recorded condition, 1 logistic
regression with all group scores, age and sex, and 1 for each group alone; `conditions_assoc_adjusted_jno.py` adds
income, education, health insurance and 2 barriers to care (eMethods 4). `breast_timing_jno.py` compares the date of the
first breast cancer code with the survey date. Their files ship in `outputs/workbench/R2025Q4R6/`:

| Folder | Workbench script | What it holds |
|---|---|---|
| `conditions_assoc_jno/` | `conditions_assoc_jno.py` | for each encoder and condition, the joint test, the number of groups significant together and alone, and every group's odds ratio together and alone |
| `conditions_assoc_adjusted_jno/` | `conditions_assoc_adjusted_jno.py` | the same with the adjustment, and each group's share of its log odds ratio kept |
| `breast_timing_jno/` | `breast_timing_jno.py` | breast cancer cases coded on or before the survey date, and after (quoted in the Results, not drawn) |

```bash
PY=env/bin/python bash scripts/run_conditions.sh
```

| Script | Writes | In the paper |
|---|---|---|
| `fig_conditions_assoc.py --encoder <encoder>` | `outputs/conditions_assoc/fig_conditions_assoc_<encoder>.{pdf,png}` | Figure 4 (gte-large-en-v1.5), eFigure 3 (the other 5) |
| `etable_conditions_jno.py` | `outputs/tables/etable_{control,assoc,breast}_jno.{tex,csv}` | eTables 9, 10 and 11 |
| `check_expected.py conditions` | (prints) | the check (6 files) |

## The other groupings (eMethods 5, eFigure 4, eTable 13)

The Discussion compares the 6 encoders with 6 other groupings of the 151 items: word counts (TF-IDF) at k = 4 and 25,
principal components and a factor model of the answers at k = 25, and multiple correspondence analysis at k = 90 and 25.
The 4 groupings estimated from the answers are fitted and scored on the Workbench (`../workbench/README.md`, steps 2, 4
and 11), and `outcome_deciles/` holds every grouping by decile.

```bash
PY=env/bin/python bash scripts/run_groupings.sh
```

| Script | Writes | In the paper |
|---|---|---|
| `etable_groupings_jno.py` | `outputs/tables/etable_groupings_jno.{tex,csv}` | eTable 13 |
| `fig_groupings_deciles.py` | `outputs/outcome_deciles/fig_groupings_deciles.{pdf,png}` | eFigure 4 |
| `check_expected.py groupings` | (prints) | the check (2 files) |

The word-count grouping is built here, from the item text only:

- `tfidf_ref.py` defines the item vectors (`vectors()`: numerals as number words, words of 2 or more characters,
  function words kept, raw counts times an IDF from the SUBTLEX-US document frequencies, unit-length rows; 151 x 586).
- `tfidf_horn.py` gives Horn's k with the encoders' settings (k = 4) and writes `outputs/k_selection/tfidf_subtlex/`:
  `env/bin/python scripts/tfidf_horn.py`.
- `build_tfidf_embeddings.py` rebuilds the item vectors and compares them with the 2 stored files in
  `data/embeddings/`; `--out <dir>` also writes them there. It reads `data/items_151.csv`, so in a fresh copy it runs
  after steps 1 and 2 of the semantic groups, and it needs the full item wording ("Item wording"):
  `env/bin/python scripts/build_tfidf_embeddings.py`.
- The basis is the leading k right singular vectors of the centered item vectors (`materialize_selection.py --svd`),
  because no sparse fit kept 99% of their variance. The Workbench code reads it from `../workbench/basis_tfidf/`, with
  the item vectors in its `item_embeddings/`.
