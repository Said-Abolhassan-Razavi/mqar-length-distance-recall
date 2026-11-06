"""
Plots accuracy against test length, with one line per model.

Each line is the model's best length 64 run, chosen by accuracy at length 64 only, so the
longer test lengths play no part in the choice. Ties go to the smaller width.
The number of key-value pairs is fixed at 4, so only the length changes.

Run from the repository root, after evaluate_length.py:
    python -m zoology.experiments.zoology_repro.plot
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from zoology.experiments.zoology_repro.configs import RESULTS_DIR
from zoology.experiments.zoology_repro.evaluate_length import TRAIN_LENGTH, get_note


MODELS = {
    "attention": ("Attention", "#2a78d6", "o"),
    "base_conv": ("BaseConv", "#eb6834", "s"),
    "based": ("Based", "#1baf7a", "^"),
}
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def select_runs(df: pd.DataFrame) -> pd.DataFrame:
    runs = df.drop_duplicates("run_id").sort_values(
        ["model", "valid_accuracy_at_train_length", "d_model"], ascending=[True, False, True]
    )
    return runs.groupby("model").head(1)


def main():
    df = pd.read_csv(RESULTS_DIR / "length_generalization.csv")
    selected = select_runs(df)
    selected[["model", "d_model", "learning_rate", "valid_accuracy_at_train_length", "run_id"]].to_csv(
        RESULTS_DIR / "plotted_runs.csv", index=False
    )
    lines = df[df.run_id.isin(selected.run_id) & (df.kv_setting == "fixed_kv")].sort_values("test_length")
    lengths = sorted(lines.test_length.unique())

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    end_labels = []
    for key, (label, color, marker) in MODELS.items():
        run = lines[lines.model == key]
        if run.empty:
            continue
        ax.plot(run.test_length, run.accuracy, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, clip_on=False, zorder=3)
        end_labels.append([label, float(run.accuracy.iloc[-1])])

    # direct labels, spread apart when lines end close together
    end_labels.sort(key=lambda item: item[1], reverse=True)
    positions = []
    for _, value in end_labels:
        positions.append(value if not positions else min(value, positions[-1] - 0.055))
    for (label, value), y in zip(end_labels, positions):
        ax.annotate(f"{label} {value:.0%}", xy=(lengths[-1], y), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9, annotation_clip=False)

    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, labels=[str(v) for v in lengths])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Test length (models trained at 64)", color=SECONDARY, fontsize=10)
    ax.set_ylabel("Recall accuracy", color=SECONDARY, fontsize=10)
    ax.set_title("Accuracy against test length, 4 key-value pairs, no retraining",
                 loc="left", color=INK, fontsize=12, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
              labelcolor=SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8, bottom=0.27)
    note = textwrap.fill("Note: " + get_note("base_conv", TRAIN_LENGTH), width=105)
    fig.text(0.02, 0.02, note, color=SECONDARY, fontsize=8, va="bottom")
    fig.savefig(RESULTS_DIR / "length_generalization.png", facecolor=SURFACE)
    print(RESULTS_DIR / "length_generalization.png")


if __name__ == "__main__":
    main()
