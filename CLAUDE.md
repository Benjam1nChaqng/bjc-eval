# CLAUDE.md — instructions for Claude Code in this repo

> **Read first**: `../bjc-portfolio/memory/INDEX.md` and any memory files relevant to the task at hand. This repo is part of the `bjc-platform` ecosystem; portfolio-level strategy and decisions live in `bjc-portfolio`.

## What this repo is

`swe-judge` is a multi-judge LLM evaluation harness for software-engineering tasks. It is **Project 2** in the bjc-platform portfolio. Its purpose is to demonstrate the eval-engineering skills that LLM evaluation roles ($100–$150/hr) pay for: rubric design, judge ensembles, inter-rater reliability measurement, and honest reporting.

The full product spec is in `PRODUCT.md` (kept in `bjc-portfolio/bjc-eval/` for the meta-repo).

## Working agreements

These are non-negotiable in this repo:

1. **TDD-first.** No new module without a passing test. The 48-test baseline must stay green. Run `pytest tests/ --no-cov` before any commit; never commit on red.
2. **Same prompt for all judges.** The system prompt in `prompts.py` is shared across Anthropic, OpenAI, and Google judges. Provider-specific prompt tuning would invalidate the inter-judge agreement metric — that's the whole point.
3. **Don't change the rubric in v1.** Rubric changes go in a new version module (`rubrics/v2.py`). Old runs' scores stay interpretable.
4. **Report measured κ honestly.** If a run comes back at κ = 0.42, the README says 0.42. Inflated numbers are the failure mode this project exists to push against. Honest negative results beat dishonest positive ones.
5. **Determinism where possible.** All judges use temperature = 0.0. Dataset version is a SHA-256 hash of the task list, recorded on every Run.
6. **Lazy-import provider SDKs.** Anthropic/OpenAI/Google imports happen inside judge constructors so contributors without all three API keys can still run the test suite.

## What "done" looks like for v0.1

- ✅ Package scaffolded with 48 passing tests
- ✅ 3 judges (Anthropic, OpenAI, Google) implementing the same Judge protocol
- ✅ Mock judge for API-free testing
- ✅ Cohen's κ + Krippendorff's α reliability layer
- ✅ SQLite persistence
- ✅ Typer CLI with `run` and `report` commands
- ⏳ 50-task golden dataset (5 seed examples shipped in `data/golden_seed.jsonl`)
- ⏳ 20-item human ground truth subset
- ⏳ First measured-numbers run published to README
- ⏳ CI green on GitHub Actions

## Architecture rules

```
src/swe_judge/
├── tasks.py          # Pydantic models — DO NOT add provider-specific fields
├── rubrics/          # Version each rubric (v1.py, v2.py, ...)
├── prompts.py        # ONE system prompt for all judges
├── judges/
│   ├── base.py       # Judge Protocol — sealed
│   ├── mock.py       # No external deps
│   ├── anthropic.py  # tool_use with forced tool_choice
│   ├── openai.py     # function_call with forced tool_choice
│   └── google.py     # function_declarations with mode=ANY
├── reliability.py    # Pure functions over Score lists
├── storage.py        # SQLite, no ORM
├── runner.py         # Thread-pool fan-out over (task × judge)
└── cli.py            # Typer
```

Rule of thumb: a change that affects judge fairness (prompt, schema, temperature, model choice) requires bumping the rubric version. A change that's purely infrastructure (storage backend, CLI flags, parallelism) does not.

## Build and test

```bash
pip install -e ".[dev]" --break-system-packages
pytest tests/ --no-cov          # 48 tests, must stay green
swe-judge --help                # smoke test
```

Smoke-test the full pipeline with mocks (no API keys needed):

```bash
python -c "
from swe_judge.tasks import Task
from swe_judge.judges.mock import MockJudge
from swe_judge.runner import run_evaluation
tasks = [Task(id='t1', category='bug_fix', difficulty=3, source='custom',
              prompt='fix', reference_solution='done')]
result = run_evaluation(tasks=tasks, judges=[MockJudge()],
                        model_outputs={'t1':'out'}, model_under_test='X')
print(result.scores)
"
```

## When you need to expand the dataset

The seed dataset is 5 tasks. Getting to 50 requires:
- 20 bug_fix (off-by-one, dedupe, edge-case handling, type confusion, threading, ...)
- 15 test_write (parsing, data structures, error paths, integration scenarios)
- 15 code_review (security, performance, API design, maintainability, compliance)

Each new task must have:
- A canonical reference solution
- Either executable test_cases (bug_fix, test_write) OR a written rubric note (code_review)
- Difficulty score 1-5 (calibrated against existing tasks)
- ULID identifier — generate with `ulid.new()`
- Tags for filtering

The 20-item human ground-truth subset should be drawn proportionally across categories.

## Things to refuse

- Adding a new judge whose prompt differs from the others. If a provider's SDK requires schema tweaks, adapt around them in the judge wrapper — the *prompt itself* stays identical.
- Adding features to v0.1 that aren't in `PRODUCT.md`. Scope discipline matters more than features.
- Caching judge outputs to disk and replaying them as if they were fresh. Reproducibility means recording the dataset version and trusting that.
- Renaming Score fields without a migration plan. The SQLite schema is part of the API.
