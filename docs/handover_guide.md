# SCRBenchmark and scRAW Handover Guide

This is the first document a new maintainer or intern should read. It explains
what is immediately reproducible from Git, where the scRAW code lives, how to
run it, and which external assets are still required for the complete M2 report
campaign.

The scientific context and reported results are available in
[`Rapport_Stage_M2_Fabien_Bidet.pdf`](Rapport_Stage_M2_Fabien_Bidet.pdf).

## 1. Handover Status

| Capability | Status from a fresh clone | Notes |
| --- | --- | --- |
| Read and modify SCRBenchmark | Ready | Main application code is under `src/scrbenchmark/`. |
| Read and modify scRAW | Ready | Public backend and presets are vendored under `vendor/scraw_inductive/`; dedicated research code is under `vendor/scraw_dedicated/`. |
| Run the Streamlit interface | Ready after dependency installation | Start it with `./run.sh`. |
| Run scRAW from the command line | Ready after dependency installation and providing a compatible H5AD file | Use the registered `scRAW` method shown below. |
| Run a small benchmark | Ready | Baron can be prepared with the setup script, or any compatible H5AD can be used. |
| Generate report job plans | Ready | Missing datasets are safely marked `blocked_missing_data`. |
| Recompute every report experiment | Requires external data | The 13 exact stable_generalist H5AD files are not stored in Git. |
| Replay the 13 preserved transductive scRAW checkpoints | Ready after dataset acquisition | Checkpoints, configurations, weights, embeddings, results, and replay code are tracked under `scraw-transductive-stable-generalist/`. |
| Reuse additional inductive checkpoints | Depends on the experiment | Some inductive bundles require external preprocessing-state and centroid artifacts. |
| Run every legacy external method | Environment-dependent | Some author implementations require legacy runtimes, notably TensorFlow 1.14 for scAIDE. |

The repository is therefore self-contained for understanding scRAW, developing
SCRBenchmark, running tested methods, and conducting new experiments. Complete
numerical reproduction of every report row additionally requires the exact
datasets and sufficient compute resources.

## 2. First Installation and Health Check

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-reproduction.txt
pip check
./scrbenchmark list-algorithms
pytest -q
```

Open the interface with:

```bash
./run.sh
```

The terminal displays the local Streamlit URL, normally
`http://localhost:8501`.

## 3. Where scRAW Lives

| Location | Purpose |
| --- | --- |
| `vendor/scraw_inductive/src/scraw/` | Self-contained public scRAW pipeline, model, trainer, presets, inference, and evaluation. |
| `vendor/scraw_inductive/configs/` | Public `default` and Baron-compatible configurations, including stable trial 0017. |
| `vendor/scraw_dedicated/src/scraw_dedicated/` | Dedicated experimentation, search, weighting, and research utilities. |
| `scripts/reproduction/adapters/run_scraw_external.py` | Adapter used by the public method registry. |
| `methods/report_methods.yaml` | Declarative `scRAW` registration used by CLI scripts and the interface. |
| `scripts/reproduction/run_scraw_leave_one_batch.py` | Inductive leave-one-batch execution. |
| `scripts/reproduction/run_scraw_from_weights.py` | Inference from an existing checkpoint and preprocessing state. |
| `scraw-transductive-stable-generalist/` | Thirteen enriched transductive checkpoints with matching configs, outputs, validation tables, and replay documentation. |
| `scripts/reproduction/replay_scraw_transductive_checkpoint.py` | Replay entry point for those enriched transductive checkpoints. |

The two public presets are:

- `default`: stable/generalist trial 0017;
- `baron`: Baron-compatible base configuration.

## 4. Run scRAW from the Command Line

Validate the generated command without training:

```bash
python scripts/reproduction/validate_method.py \
  --method scRAW \
  --data data/baron_human_pancreas.h5ad \
  --output results/scraw_validation \
  --dataset-key baron_human_pancreas \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --device auto
```

Run it through the common method entry point:

```bash
python scripts/reproduction/run_method.py \
  --method scRAW \
  --data data/baron_human_pancreas.h5ad \
  --output results/scraw_default \
  --dataset-key baron_human_pancreas \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --device auto \
  --scraw-preset default \
  --overwrite
```

Change the dataset, observation-column names, cluster count, device, and output
directory for a new experiment. Use `--scraw-preset baron` to select the other
public preset.

### Replay a preserved report checkpoint

The repository contains one enriched transductive checkpoint for each of the
13 stable-generalist datasets. For example:

```bash
python scripts/reproduction/replay_scraw_transductive_checkpoint.py \
  --checkpoint scraw-transductive-stable-generalist/model_weights/checkpoints/model_kang_pbmc_gse96583_singlets_raw_counts.pt \
  --config scraw-transductive-stable-generalist/runs/kang_pbmc_gse96583_singlets_raw_counts/seed_42/config/config_used.json \
  --data data/stable_generalist/kang_pbmc_gse96583_singlets_raw_counts.h5ad \
  --output results/replayed_scraw/kang_pbmc \
  --device auto
```

The checkpoint restores the final model and dynamic cell weights. The matching
H5AD remains necessary to recompute embeddings, clustering, metrics, and
per-cell exports.

## 5. Run scRAW and Other Experiments from Streamlit

Start with `./run.sh`, then use:

1. `Data Upload` to load and inspect an H5AD file.
2. `Data Split` and `Preprocessing` to configure the experimental protocol.
3. `Algorithm Config` and `Analysis` for built-in SCRBenchmark algorithms.
4. `Customize Benchmark` for registered report methods, including scRAW.
5. `Report Reproduction` to generate stable-generalist, inductive,
   loss-transfer, Harmony, and biological-interpretation launchers.
6. `Results Explorer` to reopen and compare saved result directories.

The interface generates commands and plans but does not bypass missing-data or
legacy-environment requirements.

## 6. Prepare Data

Prepare the smaller Baron entry point:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

For the report campaign, place the 13 exact files under
`data/stable_generalist/`, then verify checksums and AnnData metadata:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files

python scripts/reproduction/download_datasets.py --verify-only
```

The expected filenames, sizes, SHA256 checksums, dimensions, labels, and batch
columns are versioned in
`data/stable_generalist/download_manifest.csv`. Do not replace these files with
similarly named but differently preprocessed datasets if exact report numbers
are required.

## 7. Reproduce the M2 Report

Read these documents in order:

1. [`Rapport_Stage_M2_Fabien_Bidet.pdf`](Rapport_Stage_M2_Fabien_Bidet.pdf)
   for the scientific question, methods, results, and discussion.
2. [`report_reproduction_map.md`](report_reproduction_map.md) for the mapping
   between figures/tables and software campaigns.
3. [`report_reproduction_steps.md`](report_reproduction_steps.md) for the
   numbered execution procedure.
4. [`../scripts/reproduction/README.md`](../scripts/reproduction/README.md) for
   the role of each low-level script.

Once the datasets are verified, generate the plans:

```bash
python scripts/reproduction/build_stable_generalist_plan.py \
  --output-root results/stable_generalist_repro \
  --python-bin "$(which python)" \
  --device cuda

python scripts/reproduction/build_report_plan.py \
  --output-root results/report_repro \
  --python-bin "$(which python)" \
  --device cuda \
  --campaigns inductive,loss_transfer,deg
```

Inspect the generated CSV files before running the launchers. Jobs whose input
data is missing remain blocked by default.

## 8. Conduct New Experiments

For standard algorithms, use `./scrbenchmark run` or the main Streamlit flow.
For scRAW or report methods, use `run_method.py` or `Customize Benchmark`.
For a new external algorithm, follow
[`algorithm_extension_guide.md`](algorithm_extension_guide.md). For a new
dataset, follow [`dataset_integration_guide.md`](dataset_integration_guide.md).

Always keep the generated configuration, labels, embeddings, metrics, random
seed, dataset checksum, and software commit with the experiment output.

## 9. Known External Requirements

- The 13 exact stable_generalist H5AD files must be handed over separately or
  published at stable URLs.
- Long deep-learning campaigns require substantial CPU/GPU time and disk
  space; small smoke tests do not establish identical performance across all
  hardware.
- The 13 transductive scRAW checkpoints, matching configurations, cell
  weights, embeddings, result tables, and replay script are already versioned.
- Additional inductive preprocessing states, centroid references, or exact
  Baron report labels may still need to be supplied explicitly.
- Some legacy author methods need isolated environments that cannot be merged
  into the modern core environment.

These are handover assets and infrastructure requirements, not silent software
failures. The planners expose them rather than generating invalid runnable
jobs.
