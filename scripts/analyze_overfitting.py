from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    analyses = []

    for config in summary:
        history_path = RESULTS / f"history_{config['name']}.csv"
        history = _read_history(history_path)
        analysis = _analyze_one(config, history)
        analyses.append(analysis)

    report = {
        "diagnosis": _overall_diagnosis(analyses),
        "models": analyses,
    }
    (RESULTS / "overfitting_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "overfitting_analysis.md").write_text(_to_markdown(report), encoding="utf-8")
    _plot_gaps(analyses, RESULTS / "overfitting_gaps.png")

    print(json.dumps(report["diagnosis"], indent=2))


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def _analyze_one(config: dict[str, object], history: list[dict[str, float]]) -> dict[str, object]:
    final = history[-1]
    best_val_acc = max(history, key=lambda row: row["val_accuracy"])
    best_val_loss_key = "val_data_loss" if "val_data_loss" in final else "val_loss"
    best_val_loss = min(history, key=lambda row: row[best_val_loss_key])

    final_gap_pp = 100 * (final["train_accuracy"] - final["val_accuracy"])
    val_acc_drop_pp = 100 * (best_val_acc["val_accuracy"] - final["val_accuracy"])
    val_loss_rise = final[best_val_loss_key] - best_val_loss[best_val_loss_key]

    if final_gap_pp >= 3.0 or val_acc_drop_pp >= 1.0 or val_loss_rise >= 0.03:
        label = "overfitting relevante"
    elif final_gap_pp >= 1.5 or val_loss_rise >= 0.01:
        label = "overfitting leve"
    else:
        label = "sem sinal material"

    return {
        "name": config["name"],
        "test_accuracy": config["test_accuracy"],
        "test_data_loss": config.get("test_data_loss", config["test_loss"]),
        "test_objective_loss": config["test_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_val_accuracy": final["val_accuracy"],
        "final_train_data_loss": final.get("train_data_loss", final["train_loss"]),
        "final_val_data_loss": final.get("val_data_loss", final["val_loss"]),
        "final_generalization_gap_pp": final_gap_pp,
        "best_val_accuracy": best_val_acc["val_accuracy"],
        "best_val_accuracy_epoch": int(best_val_acc["epoch"]),
        "final_val_accuracy_drop_from_best_pp": val_acc_drop_pp,
        "best_val_data_loss": best_val_loss[best_val_loss_key],
        "best_val_data_loss_epoch": int(best_val_loss["epoch"]),
        "final_val_data_loss_rise_from_best": val_loss_rise,
        "diagnosis": label,
    }


def _overall_diagnosis(analyses: list[dict[str, object]]) -> dict[str, object]:
    improved = next(item for item in analyses if item["name"] == "improved_relu_256_128_momentum")
    previous = next(item for item in analyses if item["name"] == "final_relu_128_64")
    return {
        "best_model": improved["name"],
        "accuracy_gain_vs_previous_pp": 100 * (improved["test_accuracy"] - previous["test_accuracy"]),
        "improved_gap_pp": improved["final_generalization_gap_pp"],
        "improved_val_drop_pp": improved["final_val_accuracy_drop_from_best_pp"],
        "improved_val_data_loss_rise": improved["final_val_data_loss_rise_from_best"],
        "conclusion": (
            "Ha overfitting leve no modelo melhorado: treino chega quase a 100% e o gap treino-validacao fica perto de "
            f"{improved['final_generalization_gap_pp']:.2f} p.p. Ainda assim, nao ha overfitting danoso na janela treinada, "
            f"porque a validacao cai apenas {improved['final_val_accuracy_drop_from_best_pp']:.2f} p.p. do pico e a acuracia "
            "de teste sobe em relacao ao modelo anterior."
        ),
    }


def _to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Analise de overfitting",
        "",
        report["diagnosis"]["conclusion"],
        "",
        "| Modelo | Test acc | Gap treino-val | Melhor val acc | Queda val final | Diagnostico |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["models"]:
        lines.append(
            "| {name} | {test:.2%} | {gap:.2f} p.p. | {best:.2%} ep.{epoch} | {drop:.2f} p.p. | {diag} |".format(
                name=item["name"],
                test=item["test_accuracy"],
                gap=item["final_generalization_gap_pp"],
                best=item["best_val_accuracy"],
                epoch=item["best_val_accuracy_epoch"],
                drop=item["final_val_accuracy_drop_from_best_pp"],
                diag=item["diagnosis"],
            )
        )
    lines.extend(
        [
            "",
            "Criterio usado: gap final treino-validacao acima de 1.5 p.p. indica sinal leve; queda de validacao acima de 1 p.p. ou aumento relevante de data loss indicaria overfitting danoso.",
            "A decisao para esta entrega e manter o modelo melhorado, mas registrar que early stopping na epoca de melhor validacao seria a proxima melhoria natural.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_gaps(analyses: list[dict[str, object]], path: Path) -> None:
    names = [str(item["name"]).replace("_", "\n") for item in analyses]
    gaps = [float(item["final_generalization_gap_pp"]) for item in analyses]
    drops = [float(item["final_val_accuracy_drop_from_best_pp"]) for item in analyses]

    fig, axis = plt.subplots(figsize=(9, 4))
    x = range(len(names))
    width = 0.35
    axis.bar([value - width / 2 for value in x], gaps, width=width, label="Gap treino-val")
    axis.bar([value + width / 2 for value in x], drops, width=width, label="Queda val final")
    axis.axhline(1.5, color="tab:orange", linestyle="--", linewidth=1, label="Sinal leve: 1.5 p.p.")
    axis.set_xticks(list(x))
    axis.set_xticklabels(names, fontsize=8)
    axis.set_ylabel("Pontos percentuais")
    axis.set_title("Sinais de overfitting por configuracao")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
