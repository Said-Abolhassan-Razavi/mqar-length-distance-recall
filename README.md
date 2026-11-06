# Recall at Unseen Lengths and Distances on MQAR

A small reproduction and controlled extension of
[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)
(Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré).

**Question:** if a model learns multi-query associative recall (MQAR) on sequences of length 64,
does recall still work at longer lengths, and at key-to-query distances it never saw in training?

**Answer so far (one seed):** Attention and Based keep their accuracy. BaseConv loses accuracy,
and the loss comes entirely from keys further back than its filter can reach.

This repo holds only my experiment code, a small patch and my results. The models and task
come from the [Zoology repository](https://github.com/HazyResearch/zoology), which you need to
run anything here.

## Results

![Accuracy against test length](results/length_generalization.png)

All models were trained at length 64 and tested without retraining, with 4 key-value pairs at
every length. Each row is the model's best run, chosen by accuracy at length 64 only.

| Model | Width | Length 64 | Length 128 | Length 256 |
|---|---|---|---|---|
| Attention | 64 | 99.99% | 99.98% | 99.98% |
| Based | 64 | 99.52% | 99.52% | 99.48% |
| BaseConv | 128 | 93.89% | 77.90% | 66.97% |

### Accuracy by distance

Distance is the query position minus the position of its matching key. These are the same runs
at test length 256. Distances above 64 never occur in training.

| Distance | Attention | Based | BaseConv |
|---|---|---|---|
| 1-16 | 99.96% | 99.40% | 92.84% |
| 17-32 | 100.00% | 99.66% | 94.11% |
| 33-48 | 100.00% | 99.72% | 94.77% |
| 49-64 | 100.00% | 99.74% | 96.03% |
| 65-80 | 100.00% | 99.64% | 0.00% |
| 81-96 | 100.00% | 99.54% | 0.00% |
| 97-128 | 100.00% | 99.44% | 0.00% |
| 129-192 | 100.00% | 99.50% | 0.00% |
| 193-256 | 100.00% | 98.75% | 0.00% |

BaseConv's long filter only covers the training length (64), so at longer test lengths it sees
at most about that many tokens back (65 with its short convolution). This cutoff is expected from
the filter size. The more interesting part is that Attention and Based, with no position table,
work at distances they were never trained on.

### More key-value pairs

With the paper's pairing of 8 pairs at length 128 and 16 pairs at length 256, length and task
difficulty change together, so these numbers are not a pure length test.

| Model | 4 pairs, length 64 | 8 pairs, length 128 | 16 pairs, length 256 |
|---|---|---|---|
| Attention | 99.99% | 99.96% | 99.88% |
| Based | 99.52% | 98.04% | 90.43% |
| BaseConv | 93.89% | 43.62% | 11.98% |

### Reproduction

Best validation accuracy over the two learning rates, at the end of training:

| Model | L64, d64 | L64, d128 | L128, d64 | L128, d128 |
|---|---|---|---|---|
| Attention | 99.98% | 99.98% | 99.99% | 100.00% |
| Based | 99.55% | 99.52% | 99.43% | 99.62% |
| BaseConv | 55.55% | 93.45% | 10.39% | 51.38% |

Attention and Based match the paper's pattern. **BaseConv does not fully reproduce:** at length
128 and width 64 it reached about 10%, against roughly 37% read from the paper's Figure 2.
I have not found the cause yet.

## Limits

- Every number comes from one seed, so there are no error bars yet.
- The setup is smaller than the paper's: 2 of 4 learning rates, batch size 256 at length 128 to
  fit an 8 GB GPU, and at most 64 epochs. BaseConv used all 64 epochs, so it may not have converged.
- The data uses the repository's default power-law setting (0.01), while the paper's text reports
  0.1. This is one candidate for the BaseConv gap, but I have not tested it.
- I have not reviewed later papers, so I make no claim that this result is new. Later Zoology
  configs also test some models at longer lengths.

## Model setup

No model has a position table (`max_position_embeddings=0`), so every model accepts any length.
Attention and Based use the definitions in `zoology/experiments/models_repo.py`. BaseConv is the
Figure 2 model (`kernel_size=[3, -1, 3, -1]`). Full settings are in
[experiments/zoology_repro/configs.py](experiments/zoology_repro/configs.py).

## Run it

Tested on Windows 11 with Python 3.12, PyTorch 2.11 and an 8 GB GPU, at Zoology commit
`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`. Training all 24 runs took several hours.

From PowerShell, in this repo's parent folder:

```powershell
git clone -c core.autocrlf=false https://github.com/HazyResearch/zoology.git
cd zoology
git checkout 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
git apply ..\mqar-length-distance-recall\patches\windows_and_local_results.patch
Copy-Item -Recurse ..\mqar-length-distance-recall\experiments\zoology_repro zoology\experiments\

pip install torch numpy einops tqdm click pydantic pandas matplotlib transformers

python -m zoology.launch zoology/experiments/zoology_repro/configs.py
python -m zoology.experiments.zoology_repro.evaluate_length
python -m zoology.experiments.zoology_repro.evaluate_distance
python -m zoology.experiments.zoology_repro.plot
```

Results are written to `results/zoology_repro/` in the Zoology folder. Finished runs are skipped,
so training is safe to restart.

The patch makes two changes:

- It fixes a Windows crash, where `np.random.randint` overflowed the default 32-bit integer.
- It saves results to local files, so Weights & Biases is optional.

## Files

- `experiments/zoology_repro/`: run settings, the two evaluations and the plot
- `patches/windows_and_local_results.patch`: the changes to Zoology's own files
- `results/reproduction.csv`: one row per training run
- `results/length_generalization.csv`: accuracy per run, test length and number of pairs
- `results/distance_accuracy.csv`: accuracy per run, test length and distance bin
- `results/plotted_runs.csv`: the runs chosen for the plot

## Credits

The task, models and training code are from Zoology, which is under the Apache License 2.0. The
patch modifies those files and stays under that license. If you use this work, please cite Zoology:

```bibtex
@article{arora2023zoology,
  title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2312.04927},
  year={2023}
}
```

I designed and directed this study, and used an AI coding assistant (Claude Code) to write the
code and run the experiments. This is an independent project, not affiliated with the Zoology authors.
