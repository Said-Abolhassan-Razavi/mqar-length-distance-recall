"""
Tests the gap between BaseConv at length 128, width 64 and the paper's Figure 2.

Two possible causes are checked at learning rate 1e-2:
  - seed variation, with two new seeds at the setting already used
  - the power-law setting, with two runs at 0.1, the paper's text value. The Figure 2 config and
    these runs otherwise use 0.01.

Figure 2's config also uses batch size 512 at this length. That does not fit in 8 GB, so all of
these runs stay at 256 and batch size remains untested.

Writes results/zoology_repro/gap_results.csv, one row per run. Each run is also tested on the same
length 128 test set with 8 key-value pairs (power-law setting 0.01). The test records how often the
answer is one of the sequence's own values, and the accuracy expected if the model picked among
those values at random.

Run from the repository root:
    python -m zoology.launch zoology/experiments/zoology_repro/gap.py
    python -m zoology.experiments.zoology_repro.gap
"""
import pandas as pd
import torch

from zoology.config import TrainConfig
from zoology.model import LanguageModel
from zoology.experiments.zoology_repro.configs import build_configs, is_finished, RESULTS_DIR, RUNS_DIR
from zoology.experiments.zoology_repro.evaluate_length import BATCH_SIZE, get_test_data, read_summary
from zoology.experiments.zoology_repro.seeds import with_seed


BASE_RUN = "base_conv-L128-d64-lr1.0e-02"
NEW_SEEDS = [124, 125]
PAPER_POWER_A = 0.1
PAPER_POWER_SEEDS = [123, 124]


def base_config() -> TrainConfig:
    return next(c for c in build_configs() if c.run_id == BASE_RUN)


def with_power_a(config: TrainConfig, power_a: float) -> TrainConfig:
    data = config.data.model_copy(update={
        "train_configs": [s.model_copy(update={"power_a": power_a}) for s in config.data.train_configs],
        "test_configs": [s.model_copy(update={"power_a": power_a}) for s in config.data.test_configs],
    })
    return config.model_copy(update={"data": data, "run_id": f"{config.run_id}-a{power_a}"})


def build_gap_configs() -> list:
    base = base_config()
    runs = [with_seed(base, seed) for seed in NEW_SEEDS]
    for seed in PAPER_POWER_SEEDS:
        runs.append(with_seed(with_power_a(base, PAPER_POWER_A), seed))
    return runs


@torch.no_grad()
def answer_check(config: TrainConfig, data, device: str) -> dict:
    """Accuracy, and how often the answer is a value from the same sequence."""
    model = LanguageModel(config.model)
    model.load_state_dict(torch.load(RUNS_DIR / config.run_id / "model.pt", map_location=device))
    model.to(device).eval()
    preds = torch.cat([
        model(data.inputs[start:start + BATCH_SIZE].to(device)).argmax(dim=-1).cpu()
        for start in range(0, len(data), BATCH_SIZE)
    ])
    rows, positions = (data.labels != -100).nonzero(as_tuple=True)
    num_kv_pairs = config.data.train_configs[0].num_kv_pairs
    value_positions = torch.arange(1, 2 * num_kv_pairs, 2)
    answers = preds[rows, positions]
    from_sequence = (data.inputs[rows][:, value_positions] == answers.unsqueeze(-1)).any(dim=-1).float()
    return {
        "test_accuracy": (answers == data.labels[rows, positions]).float().mean().item(),
        "answers_from_sequence": from_sequence.mean().item(),
        "random_pick_accuracy": from_sequence.mean().item() / num_kv_pairs,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = base_config()
    data = get_test_data(128, base.data.train_configs[0].num_kv_pairs)
    rows = []
    for c in [base] + build_gap_configs():
        summary = read_summary(c.run_id)
        rows.append({
            "run_id": c.run_id,
            "seed": c.seed,
            "power_a": c.data.train_configs[0].power_a,
            "final_valid_accuracy": summary["final_valid_accuracy"],
            "best_valid_accuracy": summary["best_valid_accuracy"],
            "epochs_run": summary["epochs_run"],
            **answer_check(c, data, device),
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "gap_results.csv", index=False)
    print(results.to_string(index=False))


configs = [c for c in build_gap_configs() if not is_finished(c)]


if __name__ == "__main__":
    main()
