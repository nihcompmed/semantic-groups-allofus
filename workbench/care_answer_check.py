#!/usr/bin/env python3
"""care_answer_check.py -- how care.py reads the stored answers to the barrier questions on this CDR (eMethods 3).

WHY. care.py counts a barrier from the stored answer text after its last ": " (lowercased): "yes" for the 19 cost, delay
and medication questions, "no" for the usual-source question (HealthAdvice_PlaceforHealthAdvice), and "some of the
time", "most of the time" or "always" for the clinician question (HealthProviderRaceReligion_DelayedOrNoCare). The
program codebook labels the usual-source answer "There is NO place". No other download shows how version 9 stores these
answers, so this script checks whether each rule fires. It writes every stored answer to the 21 questions (1
usual source, 10 cost, 5 delayed care, 4 medication coping, 1 clinician), how
many cohort participants gave it, and how care.py reads it. It computes nothing new for the paper.

WHAT care.py DOES WITH AN ANSWER, reproduced here with care.py's own lists and its own parsing rule (imported, not
copied): a participant has the barrier when any of their stored answers to the part's questions reads as one. An answer
that does not read as a barrier, including a skip, "Prefer not to answer" or "Don't know", counts as no barrier. A
participant enters the barrier denominator (has_survey) with any stored row for any of the 21 questions.

WRITES (-> screen_out/care_answer_check/, then screen_out/care_answer_check_<CDR>.zip; aggregates only)
  care_answer_strings.csv          1 row per question and stored answer: question, part, answer (the stored text),
                                   parsed (the text after the last ": ", lowercased, as care.py reads it), reading
                                   ("barrier", "not a barrier", or "no answer" for a skip, prefer-not or don't-know),
                                   n_people (cohort participants with that stored answer)
  care_answer_check_summary.json   the counts below, each against its total, and the check against care.py's file
    n_cohort, n_has_survey          the cohort, and those with any stored answer to the 21 questions (care.py's
                                    denominator, 87,257 in the cohort)
    n_in_survey_modules             cohort participants with any stored answer in the survey module(s) the 21 questions
                                    come from (what "answered the survey" would mean), for comparison with n_has_survey
    n_part_*                        participants with each part of the barrier, by care.py's rule
    n_no_substantive_clinician      of n_has_survey, no stored answer to the clinician question that is Always, Most of
                                    the time, Some of the time or None of the time
    n_no_substantive_barrier_items  of n_has_survey, no stored answer to any of the 20 questions of "any barrier" other than a
                                    skip, prefer-not or don't-know (these participants are counted as having no barrier)
    per question: n_people_answered, n_people_more_than_1_answer
    checked_against_care_outcomes   True when the 6 per-person columns recomputed here equal care_outcomes_per_person.csv

THE CHECK. Before writing, it recomputes care.py's per-person columns (no_usual_source, any_cost_barrier,
any_delayed_care, any_cost_coping, any_access_barrier, care_avoid_provider, has_survey) from this pull and stops unless
they equal screen_out/care_outcomes_per_person.csv for every cohort participant.

THE DISSEMINATION POLICY (disclosure.py). Within a question the answer counts partition the question's respondents,
so they go through partition_mask (a count of 1 to 20 is written "<=20", and further cells are hidden until the hidden
ones sum to more than 20). Each summary count is hidden when it or its complement in its total is 1 to 20. Printed
output shows only what is written.

INPUTS   the responses file (the cohort's ids; wb_config.py), screen_out/care_outcomes_per_person.csv (care.py
         --outcomes-only), and 2 BigQuery pulls: care.py's own ds_survey query (with the survey name added), and the
         cohort participants with any answer in those surveys.
Run:  python3 care_answer_check.py 2>&1 | tee care_3_answer_check.log
"""
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import care                                                                # noqa: E402  its question lists and output file
from disclosure import MARK, SUPP, marked, pair_hidden, partition_mask     # noqa: E402  the policy
from env_versions import versions                                          # noqa: E402
from wb_config import CFG                                                  # noqa: E402
from zip_screen_aggregates import verdict                                  # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "care_answer_check")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
PARTS = {"no usual source": care.USUAL, "cost": care.COST, "delayed care": care.DELAYED,
         "medication coping": care.COPING, "clinician differs": care.MECH}
COLUMN = {"no usual source": "no_usual_source", "cost": "any_cost_barrier", "delayed care": "any_delayed_care",
          "medication coping": "any_cost_coping", "clinician differs": "care_avoid_provider"}
FREQ_BARRIER = {"some of the time", "most of the time", "always"}                         # care.py, the clinician rule
FREQ_ANSWERS = FREQ_BARRIER | {"none of the time"}
# a skip, "Prefer not to answer" or "Don't know", judged on the ANSWER only (the text after the last ": "): judged on the
# whole stored text it matched question names such as "SkippedMedToSaveMoney: No" (caught by the synthetic test)
NONANSWER = re.compile(r"^(skip|prefer\s*not.*|don'?t\s*know|dontknow)$")


def parse(answer):
    """care.py's reading of a stored answer: the text after the last ': ', lowercased and stripped."""
    return str(answer).lower().rsplit(": ", 1)[-1].strip()


def reading(part, answer):
    p = parse(answer)
    if part == "no usual source":
        hit = p == "no"
    elif part == "clinician differs":
        hit = p in FREQ_BARRIER
    else:
        hit = p == "yes"
    if hit:
        return "barrier"
    return "no answer" if NONANSWER.match(p) else "not a barrier"


def pull(ids):
    from google.cloud import bigquery
    if not CDR:
        raise SystemExit("no CDR dataset: set WORKSPACE_CDR")
    client = bigquery.Client()
    codes = ",".join("'" + x + "'" for v in PARTS.values() for x in v)
    s = client.query(f"""SELECT s.person_id, c.concept_code, s.answer, s.survey
                         FROM `{CDR}.ds_survey` s JOIN `{CDR}.concept` c ON s.question_concept_id = c.concept_id
                         WHERE c.concept_code IN ({codes})""").to_dataframe()
    s["person_id"] = s["person_id"].astype(str)
    s = s[s["person_id"].isin(set(ids))].copy()
    names = sorted(set(s["survey"].dropna().astype(str)))
    quoted = ",".join("'" + n.replace("'", "\\'") + "'" for n in names)
    m = client.query(f"""SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE survey IN ({quoted})""").to_dataframe()
    in_modules = set(m["person_id"].astype(str)) & set(ids)
    return s, names, in_modules


def per_person(s, ids):
    """care.py's per-person columns, recomputed from the pull with care.py's rules."""
    s = s.copy()
    s["val"] = s["answer"].astype(str).str.lower().str.rsplit(": ", n=1).str[-1].str.strip()     # care.py, verbatim
    yes = lambda cc: set(s[s.concept_code.isin(cc) & (s.val == "yes")].person_id)
    F = pd.DataFrame(index=pd.Index(ids, name=ID_COL))
    F["has_survey"] = F.index.isin(set(s.person_id))
    F["no_usual_source"] = F.index.isin(set(s[(s.concept_code == care.USUAL[0]) & (s.val == "no")].person_id))
    F["any_cost_barrier"] = F.index.isin(yes(care.COST))
    F["any_delayed_care"] = F.index.isin(yes(care.DELAYED))
    F["any_cost_coping"] = F.index.isin(yes(care.COPING))
    F["any_access_barrier"] = F[["no_usual_source", "any_cost_barrier", "any_delayed_care", "any_cost_coping"]].any(axis=1)
    F["care_avoid_provider"] = F.index.isin(set(s[(s.concept_code == care.MECH[0]) & (s.val.isin(FREQ_BARRIER))].person_id))
    return F


def main():
    CFG.describe("care_answer_check")
    os.makedirs(OUT, exist_ok=True)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL])
    ids = R[ID_COL].astype(str).tolist()
    N = len(ids)
    s, names, in_modules = pull(ids)
    F = per_person(s, ids)

    # ---- the check against the file every care-outcome script reads
    assert os.path.exists(care.OUTCOMES_CSV), f"{care.OUTCOMES_CSV} is missing: run `python3 care.py --outcomes-only`"
    O = pd.read_csv(care.OUTCOMES_CSV)
    O[ID_COL] = O[ID_COL].astype(str)
    O = O.set_index(ID_COL).reindex(ids)
    bad = {}
    for c in ["has_survey"] + list(COLUMN.values()) + ["any_access_barrier"]:
        mine = F[c].values.astype(bool)
        if c == "has_survey":
            theirs = O[c].fillna(False).astype(bool).values
        else:                                   # care.py leaves these empty for people outside has_survey
            theirs = O[c].fillna(0).astype(float).values.astype(bool)
        n_bad = int((mine != theirs).sum())
        if n_bad:
            bad[c] = n_bad
    assert not bad, f"this pull does not reproduce care_outcomes_per_person.csv (columns differing: {sorted(bad)})"

    has = F["has_survey"].values
    n_has = int(has.sum())

    # ---- every stored answer, its count and care.py's reading
    rows = []
    part_of = {q: p for p, qs in PARTS.items() for q in qs}
    per_q = {}
    for q in [x for v in PARTS.values() for x in v]:
        d = s[s.concept_code == q]
        k = d.groupby("person_id").size()
        per_q[q] = {"n_people_answered": int(len(k)), "n_people_more_than_1_answer": int((k > 1).sum())}
        g = d.groupby("answer")["person_id"].nunique().sort_values(ascending=False)
        counts = g.values.astype(int)
        hide, prim = partition_mask(counts)
        shown = marked(counts, hide, prim)
        for (a, _), n in zip(g.items(), shown):
            rows.append({"question": q, "part": part_of[q], "answer": a, "parsed": parse(a),
                         "reading": reading(part_of[q], a), "n_people": n})
    A = pd.DataFrame(rows)
    # a part resting on 1 question (no usual source, clinician differs) is a sum of that question's answer cells, and
    # the clinician no-answer count is the question's non-substantive cells plus the non-respondents: where the
    # question's partition hides a cell, those summary counts are hidden too, so no hidden cell is solved back
    hidden_q = {q for q in A.question.unique() if A[A.question == q].n_people.isin([MARK, SUPP]).any()}

    # ---- the summary counts
    subst_clin = set(s[(s.concept_code == care.MECH[0]) & (s["answer"].map(parse).isin(FREQ_ANSWERS))].person_id)
    barrier_qs = [x for p, v in PARTS.items() if p != "clinician differs" for x in v]
    b = s[s.concept_code.isin(barrier_qs)]
    subst_items = set(b[~b["answer"].map(parse).str.match(NONANSWER)].person_id)
    idx = pd.Index(ids)
    n_no_clin = int((has & ~idx.isin(subst_clin)).sum())
    n_no_items = int((has & ~idx.isin(subst_items)).sum())

    def shown(c, total):
        return (MARK if 0 < c <= 20 else SUPP) if pair_hidden(c, total) else int(c)

    summary = {
        "cdr": CDR,
        "question": "how care.py reads the stored answers to the 21 barrier questions",
        "n_cohort": N,
        "n_has_survey": shown(n_has, N),
        "n_in_survey_modules": shown(len(in_modules), N),
        "survey_modules": names,
        "n_part": {p: (SUPP if len(PARTS[p]) == 1 and PARTS[p][0] in hidden_q else shown(int(F[c].sum()), n_has))
                   for p, c in COLUMN.items()},
        "n_any_access_barrier": shown(int(F["any_access_barrier"].sum()), n_has),
        "n_no_substantive_clinician": SUPP if care.MECH[0] in hidden_q else shown(n_no_clin, n_has),
        "n_no_substantive_barrier_items": shown(n_no_items, n_has),
        "per_question": {q: {k: shown(v, n_has) for k, v in d.items()} for q, d in per_q.items()},
        "checked_against_care_outcomes": True,
        "parsing": "care.py: the stored answer after its last ': ', lowercased; barrier = 'yes' (cost, delayed care, "
                   "medication coping), 'no' (no usual source), some/most of the time or always (clinician differs)",
        "disclosure": "disclosure.py: answer counts within a question by partition_mask; summary counts by pair_hidden "
                      "against their totals",
        "versions": versions(),
    }

    A.to_csv(os.path.join(OUT, "care_answer_strings.csv"), index=False)
    json.dump(summary, open(os.path.join(OUT, "care_answer_check_summary.json"), "w"), indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer) else str(o))

    # ---- printed: only what is written
    print(f"\ncohort {N:,} | with any stored answer to the 21 questions (care.py's denominator) {summary['n_has_survey']} "
          f"| with any answer in {names}: {summary['n_in_survey_modules']}")
    print("checked: the per-person columns recomputed here equal care_outcomes_per_person.csv\n")
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 70, "display.width", 200):
        print(A.to_string(index=False))
    print(f"\nparts of the barrier, by care.py's rule (of {summary['n_has_survey']}): {summary['n_part']}")
    print(f"no substantive answer to the clinician question: {summary['n_no_substantive_clinician']} | "
          f"to any of the 20 questions of any barrier: {summary['n_no_substantive_barrier_items']}")

    files = ["care_answer_strings.csv", "care_answer_check_summary.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"care_answer_check_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"care_answer_check/{fn}")
    print(f"\n[care_answer_check] download -> {zpath}")


if __name__ == "__main__":
    main()
