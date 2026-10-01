# AI Use Disclosure — AI_A1_GXX

Per the assignment brief: "Generative AI may be used for explanation,
brainstorming, or debugging only when its use is disclosed. It does not
replace authorship, testing, or the ability to explain the submitted work."

**This file is a starting draft. Each member must edit it to accurately
reflect what they personally asked the tool, what they changed, and how
they tested the result. Do not submit this unedited — an assessor comparing
this file against your live verification answers is one of the explicit
checks in the rubric.**

---

## Tool used

**Tool name:** Claude (Anthropic), Sonnet model, via claude.ai
**Used by:** *(name which member(s) actually used it — be specific)*

## Purpose

Initial scaffolding of the pipeline structure (`run_all.py`, `predict.py`,
and the `src/` modules) based on the assignment PDF's stated schema and
requirements, plus the regression/classification/clustering approach
write-up in the README.

## Important prompts / requests made

*(Fill in with your group's actual prompts if you asked for anything
beyond what's described here — e.g. "explain why the regression loss
plot looks like this," "help debug a shape mismatch error," etc.)*

- Requested a complete pipeline matching the assignment's documented
  schema, folder structure, CLI contract (`run_all.py` / `predict.py`
  argument names), and artifact file list.
- Requested that the regression section use only NumPy (no sklearn
  estimator) per the assignment's explicit constraint.

## Files affected

- `src/utils.py`, `src/data_pipeline.py`, `src/regression.py`,
  `src/classification.py`, `src/clustering.py`
- `run_all.py`, `predict.py`
- `README.md`, `requirements.txt`
- `data/AI_A1_GXX.csv` — **placeholder only**, generated for testing;
  replaced with the real lecturer-issued file before submission (fill in:
  confirm this was done, and by whom)

## How the group verified the output

*(This section must be filled in honestly by your group — it's the part
the live verification will actually probe.)*

- [ ] Each member read and can explain the module(s) tied to their role
      (see the Role table in the assignment brief).
- [ ] The pipeline was run end-to-end on the **real** lecturer-issued CSV
      (not the placeholder), and the printed SHA-256 fingerprint was
      checked against the issued file.
- [ ] `predict.py` was tested with both a valid record and a deliberately
      malformed one, and the malformed case was confirmed to fail cleanly.
- [ ] Each member made at least two meaningful, individually-attributable
      commits to the file(s) tied to their role.
- [ ] The group discussed and can explain, in their own words, why
      logistic regression was chosen for classification and why k was
      selected by silhouette score for clustering.

## No AI used sections

*(If any member's role work — e.g. the UI/UX PDF design decisions, or the
contribution mapping — was done without AI assistance, state that
explicitly here, per role.)*
