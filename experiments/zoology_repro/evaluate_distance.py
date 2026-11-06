"""
Measures recall accuracy by the distance between each query and its matching key.

This uses the models trained on length 64, without retraining, on sequences with 4 key-value
pairs. Distance is the query position minus the key position, the "interaction distance" in
the Zoology paper. BaseConv's filter stops at the training length, so its accuracy should fall
sharply past that distance, while models that compare tokens directly need not.

Writes results/zoology_repro/distance_accuracy.csv, one row per run, test length and distance bin.

Run from the repository root:
    python -m zoology.experiments.zoology_repro.evaluate_distance
"""
import torch

from zoology.data.utils import DataSegment
from zoology.model import LanguageModel
from zoology.experiments.zoology_repro.configs import build_configs, is_finished, RESULTS_DIR, RUNS_DIR
from zoology.experiments.zoology_repro.evaluate_length import (
    BATCH_SIZE, TRAIN_LENGTH, get_note, get_test_data, write_csv,
)


NUM_KV_PAIRS = 4
TEST_LENGTHS = [64, 128, 256]
BIN_EDGES = [1, 17, 33, 49, 65, 81, 97, 129, 193, 257]  # each bin includes its low edge only


def query_distances(inputs: torch.Tensor, labels: torch.Tensor):
    """Returns the example index, position and key distance of every query."""
    rows, positions = (labels != -100).nonzero(as_tuple=True)
    key_positions = torch.arange(0, 2 * NUM_KV_PAIRS, 2)
    matches = inputs[rows][:, key_positions] == inputs[rows, positions].unsqueeze(-1)
    assert (matches.sum(dim=-1) == 1).all(), "each query must match exactly one key"
    return rows, positions, positions - key_positions[matches.int().argmax(dim=-1)]


@torch.no_grad()
def correct_by_query(model: LanguageModel, data: DataSegment, device: str):
    preds = torch.cat([
        model(data.inputs[start:start + BATCH_SIZE].to(device)).argmax(dim=-1).cpu()
        for start in range(0, len(data), BATCH_SIZE)
    ])
    rows, positions, distances = query_distances(data.inputs, data.labels)
    return distances, preds[rows, positions] == data.labels[rows, positions]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data = {length: get_test_data(length, NUM_KV_PAIRS) for length in TEST_LENGTHS}
    configs = [
        c for c in build_configs()
        if is_finished(c) and c.data.train_configs[0].input_seq_len == TRAIN_LENGTH
    ]

    rows = []
    for c in configs:
        model = LanguageModel(c.model)
        model.load_state_dict(torch.load(RUNS_DIR / c.run_id / "model.pt", map_location=device))
        model.to(device).eval()
        for length, data in test_data.items():
            distances, correct = correct_by_query(model, data, device)
            for low, high in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                in_bin = (distances >= low) & (distances < high)
                if not in_bin.any():
                    continue
                rows.append({
                    "model": c.model.name,
                    "d_model": c.model.d_model,
                    "learning_rate": c.learning_rate,
                    "test_length": length,
                    "distance_low": low,
                    "distance_high": high - 1,
                    "num_queries": int(in_bin.sum()),
                    "accuracy": correct[in_bin].float().mean().item(),
                    "run_id": c.run_id,
                    "note": get_note(c.model.name, TRAIN_LENGTH),
                })
                print(f"{c.run_id} L={length} distance {low}-{high - 1}: "
                      f"{rows[-1]['accuracy']:.4f} on {rows[-1]['num_queries']} queries")
    write_csv(RESULTS_DIR / "distance_accuracy.csv", rows)


if __name__ == "__main__":
    main()
