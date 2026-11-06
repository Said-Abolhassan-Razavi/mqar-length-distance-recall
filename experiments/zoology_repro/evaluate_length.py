"""
Tests the models trained on length 64 on longer sequences, without retraining.

Writes to results/zoology_repro:
  reproduction.csv          one row per training run
  length_generalization.csv one row per model, test length and number of key-value pairs

Each test length is measured two ways. "fixed_kv" keeps 4 key-value pairs, so only the
length changes. "scaled_kv" uses the paper's pairing of 4, 8 and 16 pairs at lengths
64, 128 and 256, so the task also gets harder.

Run from the repository root:
    python -m zoology.experiments.zoology_repro.evaluate_length
"""
import csv
import json

import torch

from zoology.data.multiquery_ar import MQARConfig
from zoology.data.utils import DataSegment
from zoology.model import LanguageModel
from zoology.train import compute_metrics
from zoology.experiments.zoology_repro.configs import (
    build_configs, is_finished, CACHE_DIR, RESULTS_DIR, RUNS_DIR, VOCAB_SIZE,
)


TRAIN_LENGTH = 64
TEST_SETS = [
    ("fixed_kv", 64, 4), ("fixed_kv", 128, 4), ("fixed_kv", 256, 4),
    ("scaled_kv", 64, 4), ("scaled_kv", 128, 8), ("scaled_kv", 256, 16),
]
NUM_EXAMPLES = 3_000
BATCH_SIZE = 64
TEST_SEED = 2 ** 31  # the training seeds are all below this value
BASE_CONV_NOTE = (
    "BaseConv's long filter only covers the training length ({train_length}), "
    "so at longer test lengths it sees at most about that many tokens back "
    "({reach} with its short convolution)."
)


def get_note(model: str, train_length: int) -> str:
    if model != "base_conv":
        return ""
    return BASE_CONV_NOTE.format(train_length=train_length, reach=train_length + 1)


def get_test_data(input_seq_len: int, num_kv_pairs: int) -> DataSegment:
    config = MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        num_examples=NUM_EXAMPLES,
        random_non_queries=False,
    )
    return DataSegment.from_config(
        config, cache_dir=str(CACHE_DIR), seed=TEST_SEED + input_seq_len + num_kv_pairs
    )


@torch.no_grad()
def accuracy(model: LanguageModel, data: DataSegment, device: str) -> float:
    results = []
    for start in range(0, len(data), BATCH_SIZE):
        inputs = data.inputs[start:start + BATCH_SIZE].to(device)
        labels = data.labels[start:start + BATCH_SIZE]
        preds = model(inputs).argmax(dim=-1).cpu()
        results.extend(compute_metrics(preds, labels, [{}] * len(labels)))
    return sum(r["accuracy"] for r in results) / len(results)


def read_summary(run_id: str) -> dict:
    with open(RUNS_DIR / run_id / "summary.json") as f:
        return json.load(f)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [c for c in build_configs() if is_finished(c)]

    reproduction = []
    for c in configs:
        summary = read_summary(c.run_id)
        reproduction.append({
            "model": c.model.name,
            "train_length": c.data.train_configs[0].input_seq_len,
            "d_model": c.model.d_model,
            "learning_rate": c.learning_rate,
            "final_valid_accuracy": summary["final_valid_accuracy"],
            "best_valid_accuracy": summary["best_valid_accuracy"],
            "epochs_run": summary["epochs_run"],
            "num_parameters": summary["num_parameters"],
            "state_size": summary["state_size"],
            "run_id": c.run_id,
            "note": get_note(c.model.name, c.data.train_configs[0].input_seq_len),
        })
    write_csv(RESULTS_DIR / "reproduction.csv", reproduction)

    test_data = {(length, kv): get_test_data(length, kv) for _, length, kv in TEST_SETS}
    rows = []
    for c in configs:
        if c.data.train_configs[0].input_seq_len != TRAIN_LENGTH:
            continue
        model = LanguageModel(c.model)
        model.load_state_dict(torch.load(RUNS_DIR / c.run_id / "model.pt", map_location=device))
        model.to(device).eval()
        for kv_setting, length, kv in TEST_SETS:
            score = accuracy(model, test_data[(length, kv)], device)
            rows.append({
                "model": c.model.name,
                "d_model": c.model.d_model,
                "learning_rate": c.learning_rate,
                "kv_setting": kv_setting,
                "test_length": length,
                "num_kv_pairs": kv,
                "accuracy": score,
                "valid_accuracy_at_train_length": read_summary(c.run_id)["final_valid_accuracy"],
                "run_id": c.run_id,
                "note": get_note(c.model.name, TRAIN_LENGTH) if length > TRAIN_LENGTH else "",
            })
            print(f"{c.run_id} {kv_setting} L={length} kv={kv}: {score:.4f}")
    write_csv(RESULTS_DIR / "length_generalization.csv", rows)


if __name__ == "__main__":
    main()
