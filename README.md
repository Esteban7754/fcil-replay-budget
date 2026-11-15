# Batch Composition Matters: Replay under a Controlled Update Budget in FCIL

Code and results for the paper *Batch Composition Matters: Replay under a Controlled Update Budget in
Federated Class-Incremental Learning* (ICBDAIA 2027).

The study compares ways of using a local replay buffer in federated class-incremental learning when the
starting point, the buffer contents and the number of local updates in the second task are fixed:
CIFAR-100 split into two tasks of 50 classes (A: 0–49, B: 50–99), 10 IID clients, FedAvg and ResNet-18.

This repository holds only what the paper needs: the experiment code, the exact plan that produced the
reported runs, the per-seed results and the script that draws the figures. The experiment code is
unchanged from the revision used for the runs. Comments and messages in the code are in Spanish.

## Layout

| Path | Contents |
|---|---|
| `cifar100_three_buffer.py` | One experiment: phase 1 on A, then phase 2 on B with each buffer size (0/5/10/20 %). |
| `fl_core.py` | Clients, FedAvg aggregation, local training, replay batch construction and the adapted ER-ACE loss. |
| `cifar100_sampling.py`, `cifar100_validation.py` | Class-balanced nested buffers and checks of the training contract. |
| `experiment_common.py`, `reproducibility.py` | Shared helpers, seeding and environment record. |
| `plan_experimentos.py` | Runs a plan (JSON): trains phase 1 once per seed, then every job from that checkpoint. |
| `agregar_semillas.py`, `informe_plan.py` | Aggregate seeds and write `resultados/plan_gpu/INFORME.md`. |
| `planes/plan_gpu.json` | The six jobs reported in the paper, with their seeds and arguments. |
| `resultados/plan_gpu/` | Per-seed `config.json`, `history.csv` (accuracy per round) and `results.csv`. |
| `paper/make_figures.py` | Draws Figs. 1–3 and Table 2 from `resultados/plan_gpu`. |
| `paper/figures/`, `paper/tables/` | The figures and tables as they appear in the paper. |
| `test_*.py` | Unit tests. |

## Jobs and the paper

| Job (`resultados/plan_gpu/…`) | Paper | Seeds |
|---|---|---|
| `principal_normal` | Conventional mix; no-replay row (buffer 0 %) | 42–46 |
| `principal_equilibrados` | Balanced batches, f = 0.5 | 42–46 |
| `er_ace` | Adapted ER-ACE | 42–44 |
| `fraccion_natural`, `fraccion_025`, `fraccion_075` | Ablation of the replay fraction f (Sec. 4.3) | 42–44 |

`results.csv` has one row per buffer size. The columns used in the paper are `step3_acc_a` (accuracy on A
before phase 2), `step4_acc_a`, `step4_acc_b`, `acc_all_100_classes` (ACC) and `retention_a_percent`.
All accuracies are over the 100 classes, without task identifier. Other columns (post-hoc classifier
corrections such as WA and cRT) are recorded by the code but not discussed in the paper.

## Reproduce the figures (minutes, CPU)

```bash
pip install matplotlib numpy
python paper/make_figures.py
```

## Re-run the experiments (GPU recommended)

With Docker (CIFAR-100 is downloaded at a pinned revision while building the image):

```bash
docker build -t fcil-replay-budget:gpu --build-arg TORCH_VARIANT=cuda .
docker run --gpus all -v "$PWD/resultados_nuevos:/app/resultados" fcil-replay-budget:gpu \
    python plan_experimentos.py planes/plan_gpu.json
```

Without Docker, use Python 3.13 and the pinned versions:

```bash
pip install -r requirements.lock
python plan_experimentos.py planes/plan_gpu.json --dry-run   # list the commands
python plan_experimentos.py planes/plan_gpu.json             # run them
```

`plan_experimentos.py` skips every job that already has a `results.csv`, so run it in an empty results
folder to repeat the experiments. A single run looks like:

```bash
python cifar100_three_buffer.py --controlled-replay --local-epochs 5 --rounds 5 --seed 42 \
    --buffers 0 5 10 20 --replay-mix balanced --output-dir resultados/ejemplo
```

## Tests

```bash
python -m unittest discover -p "test_*.py"
```
