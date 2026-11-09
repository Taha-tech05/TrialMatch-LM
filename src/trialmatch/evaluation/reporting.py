"""Metric bundles, markdown comparison table, confusion-matrix and training-curve figures."""
import csv
from pathlib import Path

from trialmatch.evaluation.classification import SCOPES, report_by_scope
from trialmatch.evaluation.evidence import NOTE as EVIDENCE_NOTE
from trialmatch.evaluation.evidence import evidence_report

DEFINITIONS = """## Definitions
- **Invalid outputs** are a separate prediction ("invalid"): never mapped to a valid label, always wrong, counted in the accuracy denominator.
- **Macro-F1 (present)**: mean F1 over classes with gold support > 0. **Macro-F1 (full)**: mean over the full label set of the scope; classes absent from gold get F1 = 0 (zero_division = 0). Inclusion and exclusion use their own label sets; **overall** pools both types using the union of the two label sets (so "not enough information" and "not applicable" are pooled).
- **Evidence**: per-row set P/R/F1 on sentence IDs; both empty = 1, exactly one empty = 0, invalid output = 0. Micro = pooled sentence counts, macro = mean of per-row scores. "incl." / "excl." = including / excluding rows whose gold evidence is empty.
- """ + EVIDENCE_NOTE + """
- Baseline = majority label per criterion type learned from TRAIN only, plus always-empty evidence.
- Validation split only (n small, 8 patients): differences of a few rows are within noise.
"""
TEST_DEFINITIONS_LINE = "- Test split (8 patients, evaluated once): differences of a few rows are within noise."


def build_metrics(tag: str, records: list[dict], extra: dict | None = None, split: str = "val") -> dict:
    return {"tag": tag, "split": split, "n": len(records), **(extra or {}),
            "classification": report_by_scope(records), "evidence": evidence_report(records)}


def _f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def comparison_markdown(metrics: dict[str, dict], split: str = "val", preamble: list[str] | None = None) -> str:
    tags = list(metrics)
    L = ["# Validation comparison (val split only)" if split == "val" else f"# {split.capitalize()} comparison ({split} split)", ""]
    L += list(preamble or [])
    L += ["## Overall", "",
          "| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        m, c, e = metrics[t], metrics[t]["classification"]["overall"], metrics[t]["evidence"]["overall"]
        L.append(f"| {t} | {m['n']} | {_f(c['accuracy'])} | {_f(c['macro_f1_present_classes'])} | {_f(c['macro_f1_full_label_set'])} | "
                 f"{_f(c['invalid_rate'])} | {_f(e['including_empty_gold']['micro']['f1'])} | {_f(e['including_empty_gold']['macro']['f1'])} | "
                 f"{_f(e['excluding_empty_gold']['micro']['f1'])} | {_f(e['excluding_empty_gold']['macro']['f1'])} |")
    for scope in ("inclusion", "exclusion"):
        L += ["", f"## {scope.capitalize()} only", "",
              "| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in tags:
            c, e = metrics[t]["classification"][scope], metrics[t]["evidence"][scope]
            L.append(f"| {t} | {c['n']} | {_f(c['accuracy'])} | {_f(c['macro_f1_present_classes'])} ({c['n_classes_present_in_gold']}/{c['n_classes_full']} classes) | "
                     f"{_f(c['macro_f1_full_label_set'])} | {_f(c['invalid_rate'])} | {_f(e['including_empty_gold']['micro']['f1'])} | "
                     f"{_f(e['including_empty_gold']['macro']['f1'])} | {_f(e['excluding_empty_gold']['micro']['f1'])} | "
                     f"{_f(e['excluding_empty_gold']['macro']['f1'])} |")
    L += ["", "## Per-class precision / recall / F1 (support)", ""]
    for scope in ("inclusion", "exclusion"):
        labels = list(metrics[tags[0]]["classification"][scope]["per_class"])
        L += [f"### {scope.capitalize()}", "", "| class | support | " + " | ".join(f"{t} P / R / F1" for t in tags) + " |",
              "|---|---|" + "---|" * len(tags)]
        for lab in labels:
            sup = metrics[tags[0]]["classification"][scope]["per_class"][lab]["support"]
            cells = []
            for t in tags:
                v = metrics[t]["classification"][scope]["per_class"][lab]
                cells.append(f"{_f(v['precision'], 2)} / {_f(v['recall'], 2)} / {_f(v['f1'], 2)}")
            L.append(f"| {lab} | {sup} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Evidence breakdown by gold emptiness", "",
          "| system | scope | empty-gold rows | predicted empty (correct) | non-empty-gold rows | exact match | predicted empty (missed) | macro F1 on non-empty gold |",
          "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        for scope in SCOPES:
            b = metrics[t]["evidence"][scope]["breakdown"]
            eg, ng = b["empty_gold"], b["nonempty_gold"]
            L.append(f"| {t} | {scope} | {eg['n']} | {eg.get('n_pred_empty_correct', 'n/a')} ({_f(eg.get('rate_correctly_empty'), 2)}) | "
                     f"{ng['n']} | {ng.get('n_exact_match', 'n/a')} ({_f(ng.get('rate_exact_match'), 2)}) | "
                     f"{ng.get('n_pred_empty_missed', 'n/a')} | {_f(ng.get('macro', {}).get('f1'))} |")
    L += ["", DEFINITIONS if split == "val" else DEFINITIONS.replace(DEFINITIONS.splitlines()[-1], TEST_DEFINITIONS_LINE)]
    return "\n".join(L) + "\n"


def write_confusion_csv(path: Path, conf: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold \\ predicted", *conf["pred_labels_cols"]])
        for lab, row in zip(conf["gold_labels_rows"], conf["matrix"]):
            w.writerow([lab, *row])


def plot_confusion(path: Path, conf: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    m = np.array(conf["matrix"])
    rows, cols = conf["gold_labels_rows"], conf["pred_labels_cols"]
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3, 0.9 * len(rows) + 2.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(csv_path: Path, out_path: Path, smooth: int = 5) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path).drop_duplicates(subset=["step", "split"], keep="last").sort_values("step")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tr.step, tr.loss, color="tab:blue", alpha=0.3, lw=1, label="train loss (per step)")
    ax.plot(tr.step, tr.loss.rolling(smooth, min_periods=1).mean(), color="tab:blue", lw=2, label=f"train loss ({smooth}-step mean)")
    ax.plot(va.step, va.loss, "o-", color="tab:red", lw=2, label="validation loss (per epoch)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(f"Training curves ({csv_path.stem})")
    ax.set_yscale("log")
    spe = va.step.iloc[0] / va.epoch.iloc[0] if len(va) else None
    if spe:
        sec = ax.secondary_xaxis("top", functions=(lambda s: s / spe, lambda e: e * spe))
        sec.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"train_steps": len(tr), "val_points": va[["step", "epoch", "loss"]].to_dict("records")}
