# Semantic groups of survey items in All of Us

Code for: Aggarwal M, Periwal V. *Semantic Groups of Survey Items, Screening Thresholds, and Recorded Conditions in All
of Us.

The paper scores 151 mental health and social survey items of the All of Us Research Program jointly on 27 semantic
groups. The groups are formed from the item text with a sentence encoder, before any participant's answers are used.
The paper then relates how atypical each participant's scores are to screening thresholds, recorded conditions in the
electronic health record, and sleep recorded by wearable devices.

## Contents

- `local/` runs outside the All of Us Researcher Workbench, on a CPU. It builds the semantic groups from the item text
  for 6 sentence encoders, and it draws every table and figure of the paper from the aggregate files written on the
  Workbench, which ship in `local/outputs/workbench/R2025Q4R6/`. Start with `local/README.md`.
- `workbench/` runs on the All of Us Researcher Workbench, on participant-level data. It writes the aggregate files
  that `local/` reads. Start with `workbench/README.md`.

## Data

This repository holds no participant-level data. The participant data are from the All of Us Research Program,
Registered Tier, Curated Data Repository version 9. Approved researchers can access them through the All of Us Research
Hub (https://www.researchallofus.org). The files in `local/outputs/workbench/` hold aggregate statistics only.

## Licenses

The code is released under the MIT License (`LICENSE`). The repository includes item wording only for the PHQ-9, GAD-7,
and ACE items and the stand-alone All of Us questions. The All of Us Survey Codebooks give the wording of every other
item under its item code (`local/README.md`, "Item wording"). The license does not cover the item wording, which belongs
to the owners of each questionnaire and to the All of Us Research Program. It also does not cover the sentence-encoder
models or the SUBTLEX-US word frequencies, which are downloaded separately under their own terms (`local/README.md`).

## Contact

Manu Aggarwal, National Institutes of Health (manu.aggarwal@nih.gov).
