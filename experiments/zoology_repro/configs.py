"""
Small MQAR reproduction, sized for one 8 GB GPU, plus a length generalization test.

Data settings follow paper_configs/iclr24_zoology_figure2/configs.py. Attention and Based come
from experiments/models_repo.py with the conv_mixer in mqar_example_configs/original_mqar_configs.py.
No model has a position table (max_position_embeddings=0), so every model accepts any length.

  - Attention: add_attention, a short convolution layer then a 2-head attention layer
  - Based: add_based with feature_dim 8, a short convolution layer then a linear attention layer
  - BaseConv: the Figure 2 model, a short convolution layer then a long convolution layer.
    Its long filter has l_max taps, so at longer test lengths it cannot reach further back.

Other choices:
  - lengths 64 and 128, widths 64 and 128
  - batch size at length 128 is 256 instead of 512, so that it fits in GPU memory
  - learning rates are the top two of the paper's four, as selected by
    np.logspace(-4, -2, 4)[2:] in iclr24_explore_configs/mqar_dmodel.py

Run from the repository root. Finished runs are skipped, so it is safe to restart:
    python -m zoology.launch zoology/experiments/zoology_repro/configs.py
"""
from pathlib import Path

import numpy as np

from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.models_repo import add_attention, add_based


VOCAB_SIZE = 8_192
LENGTHS = {64: dict(num_kv_pairs=4, batch_size=512), 128: dict(num_kv_pairs=8, batch_size=256)}
WIDTHS = [64, 128]
LEARNING_RATES = np.logspace(-4, -2, 4)[2:]
MODELS = ["attention", "base_conv", "based"]
BASED_FEATURE_DIM = 8  # add_based also builds 16 and 24, and Figure 2 uses 8

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "zoology_repro"
RUNS_DIR = RESULTS_DIR / "runs"
CACHE_DIR = RESULTS_DIR / "data_cache"


def build_models(input_seq_len: int) -> list:
    factory_kwargs = {"state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE}
    conv_mixer = dict(
        name="zoology.mixers.base_conv.BaseConv",
        kwargs={"l_max": input_seq_len, "kernel_size": 3, "implicit_long_conv": True},
    )
    models = add_attention([], conv_mixer, input_seq_len, factory_kwargs)
    models += [
        m for m in add_based([], conv_mixer, input_seq_len, factory_kwargs)
        if m.sequence_mixer.kwargs["configs"][1]["kwargs"]["feature_dim"] == BASED_FEATURE_DIM
    ]
    models += [
        ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=dict(
                name="zoology.mixers.base_conv.BaseConv",
                kwargs={"l_max": input_seq_len, "kernel_size": [3, -1, 3, -1]},
            ),
            max_position_embeddings=0,
            name="base_conv",
            **factory_kwargs,
        )
        for d_model in WIDTHS
    ]
    models = [m for m in models if m.d_model in WIDTHS]
    return sorted(models, key=lambda m: (m.d_model, MODELS.index(m.name)))


def build_configs(max_epochs: int = 64):
    configs = []
    for input_seq_len, setting in LENGTHS.items():
        segment = dict(
            vocab_size=VOCAB_SIZE,
            input_seq_len=input_seq_len,
            num_kv_pairs=setting["num_kv_pairs"],
            random_non_queries=False,
        )
        data = DataConfig(
            train_configs=[MQARConfig(num_examples=100_000, **segment)],
            test_configs=[MQARConfig(num_examples=3_000, **segment)],
            batch_size=setting["batch_size"],
            cache_dir=str(CACHE_DIR),
        )
        for model in build_models(input_seq_len):
            for lr in LEARNING_RATES:
                configs.append(TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=max_epochs,
                    run_id=f"{model.name}-L{input_seq_len}-d{model.d_model}-lr{lr:.1e}",
                    logger=LoggerConfig(local_dir=str(RUNS_DIR)),
                ))
    return configs


def is_finished(config: TrainConfig) -> bool:
    return (RUNS_DIR / config.run_id / "summary.json").exists()


configs = [c for c in build_configs() if not is_finished(c)]
