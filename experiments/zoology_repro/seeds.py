"""
Repeats the three plotted length 64 runs with new seeds, then summarizes all seeds.

Each new seed changes both the starting weights and the training data. The first seed (123) is
the run already picked in plotted_runs.csv, so it was chosen as the best run, while the new seeds
were not. The summary therefore keeps every seed's value, not only the mean.

Writes to results/zoology_repro:
  seed_length.csv           one row per run, test length and number of key-value pairs
  seed_distance.csv         one row per run, test length and distance bin
  seed_length_summary.csv   mean, standard deviation, lowest and highest value across seeds
  seed_distance_summary.csv the same, per distance bin

Run from the repository root, after plot.py:
    python -m zoology.launch zoology/experiments/zoology_repro/seeds.py
    python -m zoology.experiments.zoology_repro.seeds
"""
import pandas as pd
import torch

from zoology.config import TrainConfig
from zoology.model import LanguageModel
from zoology.experiments.zoology_repro.configs import build_configs, is_finished, RESULTS_DIR, RUNS_DIR
from zoology.experiments.zoology_repro.evaluate_distance import (
    BIN_EDGES, NUM_KV_PAIRS, TEST_LENGTHS, correct_by_query,
)
from zoology.experiments.zoology_repro.evaluate_length import TEST_SETS, accuracy, get_test_data


NEW_SEEDS = [124, 125]


def plotted_configs() -> list:
    run_ids = set(pd.read_csv(RESULTS_DIR / "plotted_runs.csv").run_id)
    return [c for c in build_configs() if c.run_id in run_ids]


def with_seed(config: TrainConfig, seed: int) -> TrainConfig:
    return config.model_copy(update={
        "seed": seed,
        "data": config.data.model_copy(update={"seed": seed}),
        "run_id": f"{config.run_id}-s{seed}",
    })


def build_seed_configs() -> list:
    return [with_seed(c, seed) for c in plotted_configs() for seed in NEW_SEEDS]


def summarize(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    summary = df.groupby(keys).accuracy.agg(["count", "mean", "std", "min", "max"]).reset_index()
    return summary.rename(columns={"count": "num_seeds"})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    length_data = {(length, kv): get_test_data(length, kv) for _, length, kv in TEST_SETS}
    distance_data = {length: get_test_data(length, NUM_KV_PAIRS) for length in TEST_LENGTHS}

    length_rows, distance_rows = [], []
    for c in plotted_configs() + build_seed_configs():
        if not is_finished(c):
            raise FileNotFoundError(f"{c.run_id} has not finished")
        model = LanguageModel(c.model)
        model.load_state_dict(torch.load(RUNS_DIR / c.run_id / "model.pt", map_location=device))
        model.to(device).eval()
        run = {"model": c.model.name, "d_model": c.model.d_model, "seed": c.seed, "run_id": c.run_id}

        for kv_setting, length, kv in TEST_SETS:
            score = accuracy(model, length_data[(length, kv)], device)
            length_rows.append({**run, "kv_setting": kv_setting, "test_length": length,
                                "num_kv_pairs": kv, "accuracy": score})
            print(f"{c.run_id} {kv_setting} L={length} kv={kv}: {score:.4f}")

        for length, data in distance_data.items():
            distances, correct = correct_by_query(model, data, device)
            for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                in_bin = (distances >= low) & (distances < high)
                if in_bin.any():
                    distance_rows.append({**run, "test_length": length, "distance_low": low,
                                          "distance_high": high - 1, "num_queries": int(in_bin.sum()),
                                          "accuracy": correct[in_bin].float().mean().item()})

    lengths, distances = pd.DataFrame(length_rows), pd.DataFrame(distance_rows)
    lengths.to_csv(RESULTS_DIR / "seed_length.csv", index=False)
    distances.to_csv(RESULTS_DIR / "seed_distance.csv", index=False)
    summarize(lengths, ["model", "d_model", "kv_setting", "test_length", "num_kv_pairs"]).to_csv(
        RESULTS_DIR / "seed_length_summary.csv", index=False
    )
    summarize(distances, ["model", "d_model", "test_length", "distance_low", "distance_high"]).to_csv(
        RESULTS_DIR / "seed_distance_summary.csv", index=False
    )


configs = [c for c in build_seed_configs() if not is_finished(c)]


if __name__ == "__main__":
    main()
