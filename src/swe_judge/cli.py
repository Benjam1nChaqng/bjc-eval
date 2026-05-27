"""swe-judge CLI.

Usage:
    swe-judge run --tasks data/golden_seed.jsonl --outputs outputs.jsonl --model claude-opus-4-7
    swe-judge report --db runs/results.sqlite --run RUN_ID
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from swe_judge.judges.mock import MockJudge
from swe_judge.reliability import summary
from swe_judge.runner import run_evaluation
from swe_judge.storage import Storage
from swe_judge.tasks import HumanScore, Task

app = typer.Typer(no_args_is_help=True, help="Multi-judge LLM eval harness for SWE tasks.")
console = Console()


def _load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(Task.model_validate_json(line))
    return tasks


def _load_outputs(path: Path) -> dict[str, str]:
    outs: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            outs[rec["task_id"]] = rec["output"]
    return outs


def _load_human_scores(path: Path) -> list[HumanScore]:
    rows: list[HumanScore] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(HumanScore.model_validate_json(line))
    return rows


def _build_real_judges(judge_names: list[str]) -> list:
    """Lazy-import the real judges (SDK imports are heavy)."""
    out: list = []
    for name in judge_names:
        if name.startswith("claude") or name.startswith("anthropic/"):
            from swe_judge.judges.anthropic import AnthropicJudge

            out.append(AnthropicJudge(model=name.replace("anthropic/", "")))
        elif name.startswith("gpt") or name.startswith("openai/"):
            from swe_judge.judges.openai import OpenAIJudge

            out.append(OpenAIJudge(model=name.replace("openai/", "")))
        elif name.startswith("gemini") or name.startswith("google/"):
            from swe_judge.judges.google import GoogleJudge

            out.append(GoogleJudge(model=name.replace("google/", "")))
        elif name == "mock":
            out.append(MockJudge())
        else:
            raise typer.BadParameter(f"Unknown judge: {name!r}")
    return out


@app.command()
def run(
    tasks: Path = typer.Option(..., help="Path to tasks JSONL"),
    outputs: Path = typer.Option(..., help="Path to model outputs JSONL"),
    model: str = typer.Option(..., help="Model under test identifier"),
    judges: str = typer.Option(
        "claude-opus-4-7,gpt-5.2,gemini-3-pro",
        help="Comma-separated judge model names",
    ),
    db: Path = typer.Option(Path("runs/results.sqlite"), help="Output SQLite path"),
    human_scores: Path | None = typer.Option(
        None, help="Optional human-scored JSONL for judge-vs-human metrics"
    ),
    max_workers: int = typer.Option(6, help="Thread-pool size"),
) -> None:
    """Run a full evaluation: every judge scores every task, persist to SQLite."""
    judge_list = _build_real_judges([j.strip() for j in judges.split(",") if j.strip()])
    task_list = _load_tasks(tasks)
    output_map = _load_outputs(outputs)

    console.print(
        f"[bold]Running[/] {len(task_list)} tasks × {len(judge_list)} judges = "
        f"{len(task_list) * len(judge_list)} judgments"
    )

    def _on_progress(task_id: str, judge_model: str, result_or_failure: object) -> None:
        kind = "ok" if hasattr(result_or_failure, "scores") else "fail"
        console.log(f"[{kind}] {task_id[:12]}… via {judge_model}")

    result = run_evaluation(
        tasks=task_list,
        judges=judge_list,
        model_outputs=output_map,
        model_under_test=model,
        max_workers=max_workers,
        on_progress=_on_progress,
    )

    storage = Storage(db)
    for t in task_list:
        storage.insert_task(t)
    storage.insert_run(result.run)
    storage.insert_scores(result.scores)
    if human_scores:
        storage.insert_human_scores(_load_human_scores(human_scores))

    metrics = summary(
        result.scores,
        storage.all_human_scores() if human_scores else None,
    )
    _print_summary(metrics, n_failures=len(result.failures), run_id=result.run.id)


@app.command()
def report(
    db: Path = typer.Option(..., help="SQLite database path"),
    run_id: str = typer.Option(..., "--run", help="Run ID to summarise"),
) -> None:
    """Print reliability metrics for a stored run."""
    storage = Storage(db)
    scores = storage.scores_for_run(run_id)
    humans = storage.all_human_scores()
    if not scores:
        console.print(f"[red]No scores found for run {run_id}[/]")
        raise typer.Exit(code=1)
    metrics = summary(scores, humans if humans else None)
    _print_summary(metrics, n_failures=0, run_id=run_id)


def _print_summary(metrics: dict[str, object], n_failures: int, run_id: str) -> None:
    table = Table(title=f"Run {run_id[:12]}…", show_header=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Scores", str(metrics.get("n_scores", "—")))
    table.add_row("Failures", str(n_failures))
    mean_kappa = metrics.get("mean_inter_judge_kappa", float("nan"))
    table.add_row("Mean inter-judge κ", f"{mean_kappa:.3f}" if isinstance(mean_kappa, float) else str(mean_kappa))
    if "mean_judge_vs_human_kappa" in metrics:
        mean_jh = metrics["mean_judge_vs_human_kappa"]
        table.add_row(
            "Mean judge-vs-human κ",
            f"{mean_jh:.3f}" if isinstance(mean_jh, float) else str(mean_jh),
        )
    alpha_key = "krippendorff_alpha_with_human" if "krippendorff_alpha_with_human" in metrics else "krippendorff_alpha"
    alpha = metrics.get(alpha_key, float("nan"))
    table.add_row(
        "Krippendorff α",
        f"{alpha:.3f}" if isinstance(alpha, float) else str(alpha),
    )
    console.print(table)

    if isinstance(metrics.get("inter_judge_pairwise"), dict):
        console.print("\n[bold]Pairwise κ[/]")
        for pair, k in metrics["inter_judge_pairwise"].items():  # type: ignore[union-attr]
            console.print(f"  {pair}: {k:.3f}")


if __name__ == "__main__":
    app()
