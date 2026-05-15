# swe-judge

> Multi-judge LLM evaluation harness for software-engineering tasks. A 3-judge ensemble (Claude Opus 4.7, GPT-5.2, Gemini 3 Pro) scores model outputs across a 3-dimension rubric with worked anchors, with inter-rater reliability measured against human ground truth.

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

`swe-judge` is a reference implementation of a production-grade LLM-as-judge evaluation harness for software-engineering tasks. It demonstrates the design patterns that LLM evaluation teams at frontier labs use in practice:

- **3-judge ensemble** across three frontier labs (Anthropic, OpenAI, Google) — diversity is the point. Single-judge eval work has known correlated-error problems documented in Ye et al., ICLR 2025 ("Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge").
- **Rubric with worked anchors at every score level.** No vague "rate 1–5" prompts. Every level (1, 2, 3, 4, 5) has concrete language a judge must cite when scoring.
- **Evidence-first scoring.** Each judge must produce a rationale + anchor citation *before* the numeric score, enforced by forced tool_use / function calling.
- **Cohen's κ + Krippendorff's α** for inter-rater reliability, measured both pairwise across judges and against human ground truth on a 20-item subset.
- **Contamination-free dataset.** SWE-bench Verified scores have been shown to inflate by ~35 percentage points relative to SWE-bench Pro (the contamination-resistant variant). Our 50 tasks are novel, not drawn from any benchmark in major training corpora.

## v0.1 headline numbers

> _Numbers below are placeholders. Replace after the first full run, including the actually-measured κ — even if it's bad. Honest negative results beat inflated ones._

- **50 golden examples** across bug fixing (20), test writing (15), code review (15)
- **3 judges** × **3 dimensions** = 9 scores per task = 450 judgments per run
- **Mean inter-judge Cohen's κ:** `0.__`
- **Mean judge-vs-human Cohen's κ** (20-item subset): `0.__`
- **Krippendorff's α** (4-rater ordinal, judges + human): `0.__`
- Full run wall-clock: `~__ min`

Interpret κ via Landis & Koch (1977): 0.0–0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 **substantial**, 0.8–1.0 almost perfect.

## Architecture

```
┌─────────────┐    ┌────────────────────┐    ┌─────────────────┐
│  golden     │    │  3-Judge Ensemble  │    │  Reliability    │
│  dataset    │───>│ ─────────────────  │───>│  Cohen's κ      │
│  (50 tasks) │    │  Opus 4.7          │    │  Krippendorff α │
│             │    │  GPT-5.2           │    │                 │
│ + model     │    │  Gemini 3 Pro      │    └─────────────────┘
│   outputs   │    │  (same prompt)     │
└─────────────┘    └────────────────────┘
```

All three judges:
- Receive the **same system prompt** (built from `prompts.py`)
- Use **temperature = 0** (this is scoring, not creative writing)
- Are **forced** into structured tool_use / function calling — no free-text JSON parsing
- Must produce **rationale + anchor citation** before the score (evidence-first)

## Install

```bash
# With uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

For real LLM calls you also need API keys:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

## Usage

### Run an evaluation

```bash
swe-judge run \
    --tasks data/golden_50.jsonl \
    --outputs my-model-outputs.jsonl \
    --model claude-opus-4-7 \
    --judges claude-opus-4-7,gpt-5.2,gemini-3-pro \
    --human-scores data/human_ground_truth.jsonl \
    --db runs/results.sqlite
```

The `outputs` JSONL is the model-under-test's responses, one per task:

```json
{"task_id": "01HX...", "output": "def median(numbers):\n    ..."}
```

### Print reliability metrics

```bash
swe-judge report --db runs/results.sqlite --run RUN_ID
```

## The rubric (v1)

Each dimension is scored 1–5. Every level has a worked anchor (full text in `src/swe_judge/rubrics/v1.py`):

**Correctness** — Does the code do what was asked, including edge cases?

**Code Quality** — Idiomatic, readable, appropriately abstracted code.

**Reasoning** — Quality of the explanation accompanying the code or review.

## What's measured, what's not

**v0.1 measures:**
- Cohen's κ pairwise across judges
- Cohen's κ between each judge and a 20-item human ground truth
- Krippendorff's α treating all raters as one ordinal panel

**v0.1 does NOT measure (deferred to v0.4):**
- Position bias (which solution shows first)
- Self-enhancement bias (a model judging its own output)
- Verdict balance in the few-shot calibration
- Self-consistency (running each judge N times for median)

The deferred items are real and documented (Ye et al. 2025, Zheng et al. 2024). Their absence in v0.1 is a known limitation, not a hidden one.

## Honesty rule

If a measurement comes in worse than the targets above, this README reports the actual number. Inflated benchmark numbers are the failure mode this project exists to push against.

## References

- Ye, J. et al. (2025). *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.* ICLR 2025.
- Zheng, L. et al. (2024). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks.
- Tan, S. et al. (2024). *JudgeBench: A Benchmark for Evaluating LLM-based Judges.* arXiv:2410.12784.
- Landis, J.R. & Koch, G.G. (1977). *The Measurement of Observer Agreement for Categorical Data.* Biometrics.
- UK AI Security Institute (2024+). [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — the eval framework this project's API is designed to be compatible with.

## License

MIT. See `LICENSE`.

## Owner

Benjamin Chang · [github.com/Benjam1nChaqng](https://github.com/Benjam1nChaqng)
