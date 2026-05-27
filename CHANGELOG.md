# Changelog

All notable changes to swe-judge are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2
- Position-bias randomization in judge prompts
- Self-consistency: each judge runs N=3 times, median used
- Verdict-balanced few-shot calibration examples
- Hugging Face dataset publish workflow
- GitHub Actions CI gate: block PR if κ regresses > 0.05

## [0.1.1] — 2026-05-15

### Fixed
- Declared `pandas>=2.0` as an explicit dependency. `simpledorff` (Krippendorff's α library) requires pandas internally, and `reliability.py` also imports it directly inside the Krippendorff helper. Without this declaration, fresh installs failed 7 reliability tests with `ModuleNotFoundError: No module named 'pandas'`.

## [0.1.0] — 2026-05-15

Initial scaffold of the multi-judge eval harness.

### Added
- Pydantic v2 data models: `Task`, `Run`, `Score`, `HumanScore`, `JudgmentResult`, `DimensionScore`, `TestCase`
- Rubric v1: three dimensions (correctness, code_quality, reasoning) with worked anchors at every level 1–5
- Shared judge system prompt enforcing evidence-first scoring (rationale + anchor citation before numeric score)
- `Judge` protocol with concrete implementations:
  - `AnthropicJudge` — Claude Opus 4.7 with forced `tool_use`
  - `OpenAIJudge` — GPT-5.2 with forced function-call `tool_choice`
  - `GoogleJudge` — Gemini 3 Pro with `function_declarations` mode=ANY
  - `MockJudge` — deterministic, API-free for testing
- Single canonical `JUDGE_TOOL_SCHEMA` reused across all three providers (fairness property)
- Reliability layer: pairwise Cohen's κ, mean inter-judge κ, judge-vs-human κ, Krippendorff's α
- SQLite storage with FK constraints and indexed score lookups
- Thread-pool runner: parallel fan-out over (task × judge) pairs with non-fatal failure collection
- Typer CLI: `swe-judge run` and `swe-judge report`
- 48 tests covering models, rubric, mock judge, reliability math, storage, and runner orchestration
- 5-task seed dataset (`data/golden_seed.jsonl`): 2 bug_fix, 2 test_write, 1 code_review

### Notes
- v0.1 deliberately omits bias mitigations covered in Ye et al. (ICLR 2025) — those land in v0.2. The omission is documented, not hidden.
- Tests use varied-value fixtures because Cohen's κ is mathematically undefined when score variance is zero. Fixed during scaffolding.
