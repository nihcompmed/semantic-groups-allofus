# Local code

This folder holds everything that runs outside the All of Us Researcher Workbench:

- **The semantic groups** (`semantic_groups/`, run by `run_semantic_groups.sh`). They are built from the text of the
  151 survey items, for 6 sentence encoders, with gte-large-en-v1.5 specified in advance as the main one. This part
  uses the item text and the stored item embeddings only, never a participant's answers.
- **The paper** (`paper/`, run by `run_paper.sh`): one script per figure, appendix table, supplementary file and figure
  source data, named after it. They are drawn from aggregate files written on the Researcher Workbench, which ship in
  `outputs/workbench/R2025Q4R6/`. The code that writes those files is in `../workbench/`, whose `README.md` gives the
  steps in order.

Everything here runs on a CPU. No participant-level data are in this folder.

## Layout

```
README.md               this file
run_semantic_groups.sh  the semantic groups, every step in order, then check.py groups
run_paper.sh            every script of paper/, then check.py paper
check.py                compares a run with expected/
semantic_groups/        the steps of the semantic groups
paper/                  one script per display item
ross_proj/              the fitting library (parallel analysis and the sparse fit)
data/                   the inputs ("Inputs" below)
outputs/                the semantic groups of record and the Workbench downloads, read by paper/; the runs write here
expected/               the files of record, compared with each run by check.py
```

`run_paper.sh` writes the paper's figures, tables and workbooks into `../output/`, beside this folder.

## Environment

Python 3.11 with these library versions:

```bash
python3.11 -m venv env
env/bin/pip install numpy==2.4.6 scipy==1.17.1 pandas==2.3.3 scikit-learn==1.9.0 matplotlib==3.11.0 \
    statsmodels==0.14.6 openpyxl==3.1.5 jax==0.10.1 jaxlib==0.10.1 optax==0.2.8 tqdm==4.68.2
```

The run scripts set `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu` themselves. Both are started from this folder as
`PY=env/bin/python bash <run>.sh` and end with `check.py`, which compares the result with `expected/` and prints
`N of N match the files of record` when they are the same.

## The paper

```bash
PY=env/bin/python bash run_paper.sh
```

The run takes a few minutes. It reads only the files already in `outputs/`, so it does not need the semantic groups to
be rebuilt first. Each script's docstring states what it draws, what it reads and what it writes.

Each script also runs on its own, to see how one display item is drawn. It writes into `../output/` (beside this
folder), each figure as `.png` and `.pdf`, under the names below. The manuscript's own copies are in `expected/output/`
under the same names, so a figure can be compared with the one in the paper, for example:

```bash
cd paper
python figure4.py
cmp ../../output/figures/outliers_factor.png ../expected/output/figures/outliers_factor.png
```

`cmp` prints nothing when the 2 files are the same, byte for byte. `python check.py paper` (from this folder) compares
every figure, table and workbook at once.

| In the paper | Script (`paper/`) | Writes (`../output/`) |
|---|---|---|
| Figure 1 | `figure1.py` | `figures/method` |
| Figure 2 | `figure2.py` | `figures/deciles` |
| Figure S3 | `figure2_supplement1.py` | `figures/deciles_conditions_encoders` |
| Figure S4 | `figure2_supplement2.py` | `figures/deciles_age_sex` |
| Figure 3 | `figure3.py` | `figures/contributors` |
| Figure 4 and Figures S5 and S6 | `figure4.py` | `figures/outliers_factor`, `outliers_factor_factors`, `outliers_factor_groups` |
| Figure 5 | `figure5.py` | `figures/below_threshold` |
| Figure S7 | `figure5_supplement1.py` | `figures/below_threshold_encoders` |
| Figure 6 and Figures S8 and S9 | `figure6.py` | `figures/below_outliers_factor`, `below_outliers_factor_groups`, `below_outliers_factor_factors` |
| Figure 7 and Figures S10 to S14 | `figure7.py` | `figures/conditions`, then `conditions_<encoder>` for bge-m3, bge-large-en-v1.5, all-mpnet-base-v2, e5-large-v2, qwen3-embedding-0.6b |
| Figures 2 to 7 Source Data | `figure2_source_data.py` to `figure7_source_data.py` | `supplementary/Figure<n>_source_data.xlsx` |
| Figure S1 | `appendix2_figure1.py` | `figures/k_choice` |
| Figure S2 | `appendix2_figure2.py` | `figures/sparsification` |
| Tables S1 and S2 | `appendix2_table1.py`, `appendix2_table2.py` | `tables/app_groups.tex`, `app_switch.tex` |
| Table S3 | `appendix3_table1.py` | `tables/app_thresholds.tex` |
| Table S4 | `appendix4_table1.py` | `tables/app_factors.tex` |
| Table S5 | `appendix5_table1.py` | `tables/app_factor_conditions.tex` |
| Supplementary Files 1 to 3 | `supplementary_file1.py`, `supplementary_file2.py`, `supplementary_file3.py` | `supplementary/Supplementary_file_<n>.xlsx` |

Shared by these scripts:

| File | What it holds |
|---|---|
| `common.py` | the folders, the encoders, the 8 recorded conditions, the printed names of the constructs, the rule for which item wording may be reprinted, the figure and table writers |
| `workbook.py` | the workbook writer, the Holm adjustment, the number formats and the names of the groupings |

Some scripts also use another's functions: `figure5.py` uses `figure2.py`'s series and labels, `figure6.py` uses
`figure4.py`'s style and helpers, `figure5_source_data.py` uses `figure2_source_data.py`'s sheets,
`figure6_source_data.py` uses `figure4_source_data.py`'s, `appendix5_table1.py` and `supplementary_file1.py` use
`appendix4_table1.py`'s factors, `supplementary_file1.py` uses `appendix2_table2.py`'s crossing counts, and
`appendix2_figure2.py` uses the core rule of `semantic_groups/step8_cores.py`.

## The semantic groups

```bash
PY=env/bin/python bash run_semantic_groups.sh
```

The run takes about 30 to 40 minutes on a 48-core CPU: step 3 about 7 minutes, step 5 about 20 minutes (600 fits of
about 10 seconds each, 6 jobs in parallel). It ends with `check.py groups`. `step5_random_starts.py` skips any start
whose files already exist; to repeat step 5 from scratch, delete `outputs/random_starts/` first.

| Step | Script (`semantic_groups/`) | Reads | Writes | In the paper |
|---|---|---|---|---|
| 1 | `step1_text.py` | `data/sources/administered_text_v9.csv`, `core_items_151.csv` | `data/sources/items_text_of_record.csv` | the text encoded (Appendix S2) |
| 2 | `step2_items.py` | `data/sources/*`, `data/encoding_map_all151.csv` | `data/items_151.csv` | the item table |
| 3 | `step3_k.py` | the embeddings | `outputs/k_selection/` | the number of groups, by Horn's parallel analysis |
| 4 | `step4_fit.py` | the embeddings, k | `outputs/fits/` | the fit started from the principal components |
| 5 | `step5_random_starts.py` | the embeddings, k | `outputs/random_starts/<encoder>__k<k>_ls1.0/` | 100 random orthonormal starts per encoder |
| 6 | `step6_select.py` | steps 4 and 5 | `outputs/random_starts/selection.json`, `candidates.csv` | the selection rule (Appendix S2) |
| 7 | `step7_basis.py` | `selection.json` | `outputs/basis_fit/<encoder>/` | the basis of each encoder |
| 8 | `step8_cores.py` | the basis, `data/items_151.csv` | `outputs/cores/` | the core items of every group (Appendix S2) |
| 9 | `step9_package.py` | the basis, the cores | `outputs/basis/` | the package the Workbench code reads (a copy is in `../workbench/basis/`) |
| 10 | `step10_agreement.py` | the cores | `outputs/xenc/` | agreement across encoders (Appendix S2) |
| 11 | `../paper/appendix2_table1.py`, `appendix2_table2.py` | `outputs/basis/components.csv`, `data/items_151.csv` | `outputs/tables/`, `../output/tables/` | Tables S1 and S2 |
| 12 | `../check.py groups` | `outputs/`, `expected/` | (prints) | the check |

`semantic_groups/settings.py` holds the folders, the encoder list and the embedding loader the steps share. Three more
scripts are checks, not steps of the run: `embed.py` encodes the item text again ("Encoding the item text again"
below), and `tfidf_k.py` and `tfidf_embeddings.py` give the word-count grouping's k and rebuild its stored vectors, from
the definition in `tfidf.py` ("Item wording" below).

### Inputs

In `data/`:

- `sources/administered_text_v9.csv`: the question string of each item as administered, from the survey metadata of
  the All of Us Curated Data Repository, version 9. It holds no participant data.
- `sources/core_items_151.csv`: each item's concept ID, survey, and published wording.
- `sources/construct_map_151.csv`: each item's questionnaire and documented construct.
- `sources/scales_prism.csv`: each item's response range and reverse coding.
- `encoding_map_all151.csv`: the numeric value of every answer. The Workbench code uses it too, and step 2 checks the
  response ranges against it.
- `standalone_sources.csv`: the source study of each stand-alone All of Us question.
- `condition_test_codes.csv`: the codes of the recorded tests that define the populations for hypercholesterolemia,
  breast cancer and prostate cancer, as `../workbench/step5_outcomes.py` lists them, with their concept names
  (Supplementary File 2).
- `embeddings/admin151_items_<encoder>.npz`: the item embeddings. Each holds 151 unit-length vectors of 768
  (all-mpnet-base-v2) or 1024 numbers, computed by `embed.py` from `text_of_record`.
- `embeddings/admin151_items_tfidf-subtlex-k4.npz` and `..._k25.npz`: the word-count (TF-IDF) item vectors, 151 x 586,
  the same vectors under 2 names (Appendix S4).

### Item wording

The wording of an item is included only where its questionnaire's terms permit redistribution: the PHQ-9, GAD-7, and
ACE items and the 13 stand-alone All of Us questions. For the other items, the text columns of `data/sources/*.csv`,
`data/items_151.csv`, and `../workbench/items_151.csv` are empty. The All of Us Survey Codebooks
(https://support.researchallofus.org/hc/en-us/articles/360051991531) give the wording of every item under its item
code, the `item` column.

Every step of the run works without that wording, because every step after the item table reads the stored
embeddings. Two optional checks need the full wording, entered in `data/sources/administered_text_v9.csv` and
`data/sources/core_items_151.csv`: encoding the item text again (below) and rebuilding the word-count vectors with
`tfidf_embeddings.py`.

The word-count scripts (`tfidf.py`, `tfidf_k.py`, `tfidf_embeddings.py`) need the SUBTLEX-US word frequencies
(Brysbaert and New 2009), which are not redistributed here. Download
https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip and put the file
`SUBTLEXus74286wordstextversion.txt` (md5 `ee7be8db21f26e1925ac0ea3ab598f32`) in `data/subtlexus/`, or pass its path
with `--subtlex`. `tfidf_k.py` gives the word-count grouping's k (`outputs/k_selection/tfidf_subtlex/`), and
`tfidf_embeddings.py` rebuilds the 2 stored word-count files and compares them with `data/embeddings/`.

### Encoding the item text again (optional)

The run starts from the stored embeddings. Encoding the text again is a check, not a step of the run, and it needs the
full item wording. On a CPU, 4 encoders reproduce the stored vectors bit for bit, gte-large-en-v1.5 at a cosine
similarity of 1.000000 and Qwen3-Embedding-0.6B at 0.9998 or more.

| Encoder | Hub name | Revision |
|---|---|---|
| bge-m3 | BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 |
| bge-large-en-v1.5 | BAAI/bge-large-en-v1.5 | d4aa6901d3a41ba39fb536a557fa166f842b0e09 |
| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf5434bc2275fc9bab85f82640a19130 |
| e5-large-v2 | intfloat/e5-large-v2 | f169b11e22de13617baa190a028a32f3493550b6 |
| gte-large-en-v1.5 | Alibaba-NLP/gte-large-en-v1.5 | 104333d6af6f97649377c2afbde10a7704870c7b |
| qwen3-embedding-0.6b | Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |

Two environments, both with torch 2.12.0 (CPU): sentence-transformers 5.5.1 and transformers 5.10.2 for 5 encoders, and
sentence-transformers 3.4.1 and transformers 4.49.0 for gte-large-en-v1.5, whose published custom code
(Alibaba-NLP/new-impl, revision 40ced75c3017eb27626c9d4ea981bde21a2662f4) does not run under transformers 5. Download
each model into a folder named after the encoder (for example `models/bge-m3/`), then, from `semantic_groups/`:

```bash
CUDA_VISIBLE_DEVICES="" ENV5/bin/python embed.py --model bge-m3 --models-dir ../models --out /tmp/bge-m3.npz
CUDA_VISIBLE_DEVICES="" ENV4/bin/python embed.py --model gte-large-en-v1.5 --models-dir ../models --trust-remote-code --out /tmp/gte.npz
```

Without `--out`, `embed.py` overwrites the stored embeddings in `data/embeddings/`.

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
the paper are the core items in `outputs/basis/components.csv`, so `check.py groups` compares that file on its own. The
Workbench code reads the copy of the basis package in `../workbench/basis/`.
