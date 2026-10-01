# AI Use Disclosure — AI_A1_G02

Per the assignment brief: "Generative AI may be used for explanation,
brainstorming, or debugging only when its use is disclosed. It does not
replace authorship, testing, or the ability to explain the submitted work."

---

## Tool used

**Tool name:** Gemini, Claude (Anthropic)
**Used by:** Used As Group.

## Purpose

Initial scaffolding of the pipeline structure (`run_all.py`, `predict.py`,
and the `src/` modules) like you said on the assignment PDF's stated schema and
requirements, plus debugging terminal setup, pathing issues, JSON string formatting, and documentation template population (`TEST_LOG.md`, `README.md`).

## Important prompts / requests made

- Requesting help on how to structure the project schema, folder hierarchy, CLI contracts (`run_all.py` / `predict.py` argument names), and required artifact generation list.
- Requesting that the regression module use pure NumPy gradient descent from first principles (no sklearn estimator) per the assignment constraint.
- Troubleshooting shell-specific execution errors (Command Prompt path resolution vs. Git Bash JSON quote escaping) and verifying SHA-256 dataset checksums.

## Files affected

- `src/utils.py`, `src/data_pipeline.py`, `src/regression.py`, `src/classification.py`, `src/clustering.py`
- `run_all.py`, `predict.py`
- `README.md`, `requirements.txt`, `TEST_LOG.md`, `AI_USE.md`
- `data/AI_A1_GXX.csv`

## How the group verified the output

Each member thoroughly reviewed and step-debugged the generated code line-by-line to verify logic correctness, manually validated the SHA-256 fingerprint against `data_report.json`, verified output metrics across generated artifact files, and confirmed that the full pipeline executes cleanly from a fresh virtual environment.

But Honestly it's still now that clear. it would be much helpful if yu watch us through n class how how exactly to do all this 
since it's much interesting and helpful as much as it is so real. thanks.

## No AI used sections

- Data preprocessing logic verification and manual inspection of raw CSV anomalies.
- Final dataset integrity hashing and file path adjustments.
- Execution, verification, and recording of actual output test results within `TEST_LOG.md`.