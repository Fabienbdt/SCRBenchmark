"""
Customize Benchmark Page.

Allows users to define multiple benchmark configurations and generate CLI scripts
without executing them immediately.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import copy
from datetime import datetime
from pathlib import Path
import re
import shlex
import sys

from core.algorithm_registry import AlgorithmRegistry
from gui.algorithm_config import _generate_cli_command

try:
  from gui.protocol_designer import render_protocol_registry_panel, render_protocol_workbench
except Exception:
  render_protocol_registry_panel = None
  render_protocol_workbench = None

try:
  from protocols.registry import get_protocol_spec, protocol_to_customize_configs
except Exception:
  get_protocol_spec = None
  protocol_to_customize_configs = None

REPO_ROOT = Path(__file__).resolve().parents[3]
REPRODUCTION_SCRIPTS = REPO_ROOT / "scripts" / "reproduction"
if str(REPRODUCTION_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(REPRODUCTION_SCRIPTS))

try:
  from manual_protocols import ManualProtocolConfig, build_jobs, parse_seeds
except Exception:
  ManualProtocolConfig = None
  build_jobs = None
  parse_seeds = None


REPORT_DATASET_TABLE = REPO_ROOT / "reproducibility" / "stable_generalist" / "stable_generalist_dataset_table.csv"
REPORT_DATA_ROOT = Path("data") / "stable_generalist"

COMMON8_DATASET_KEYS = [
  "bbag094_zeisel",
  "bbag094_spleen",
  "baron_human_pancreas",
  "gse112013_human_testis_raw_counts",
  "kang_pbmc_gse96583_singlets_raw_counts",
  "macaque_retina_gse118480_bipolar_raw_counts",
  "paul15_bone_marrow_raw_counts",
  "Tabula_Muris_liver_filtered_raw_counts",
]

LOSS_TRANSFER_DATASET_KEYS = [
  "baron_human_pancreas",
  "bbag094_zeisel",
  "bbag094_spleen",
  "kang_pbmc_gse96583_singlets_raw_counts",
  "paul15_bone_marrow_raw_counts",
]

REPORT_PRIMARY_METHODS = [
  "scRAW",
  "scAIDE",
  "CellSIUS",
  "DeepScena",
  "scCAD",
  "GiniClust",
  "scvi",
  "scMAE",
  "pca_leiden",
  "scNAME",
  "Harmony",
  "ComBat",
  "DESC",
  "Scanorama",
]

REPORT_HARMONY_METHODS = [
  "Harmony",
  "scMAE+Harmony",
  "scNAME+Harmony",
  "scvi+Harmony",
  "DeepScena+Harmony",
  "CellSIUS+Harmony",
  "GiniClust+Harmony",
  "scAIDE+Harmony",
  "scCAD+Harmony",
]

LOCAL_BARON_COMPARISON_ALGORITHMS = [
  "pca_kmeans",
  "pca_leiden",
  "scdeepcluster",
  "sccdcg",
  "sc_mae",
  "scname",
]

INDUCTIVE_ALGORITHMS = ["scraw", "scname", "sc_mae", "scdeepcluster", "scaide", "pca_harmony"]

INDUCTIVE_SPLIT_PRESETS = [
  {
    "dataset_key": "baron_human_pancreas",
    "name": "h234_to_h1",
    "split_key": "batch",
    "train_batches": "human2,human3,human4",
    "test_batches": "human1",
  },
  {
    "dataset_key": "baron_human_pancreas",
    "name": "h134_to_h2",
    "split_key": "batch",
    "train_batches": "human1,human3,human4",
    "test_batches": "human2",
  },
  {
    "dataset_key": "baron_human_pancreas",
    "name": "h124_to_h3",
    "split_key": "batch",
    "train_batches": "human1,human2,human4",
    "test_batches": "human3",
  },
  {
    "dataset_key": "baron_human_pancreas",
    "name": "h123_to_h4",
    "split_key": "batch",
    "train_batches": "human1,human2,human3",
    "test_batches": "human4",
  },
  {
    "dataset_key": "bbag094_spleen",
    "name": "3F56_to_3M8",
    "split_key": "batch",
    "train_batches": "3-F-56",
    "test_batches": "3-M-8",
  },
  {
    "dataset_key": "gse112013_human_testis_raw_counts",
    "name": "donor12_to_donor3",
    "split_key": "batch",
    "train_batches": "Donor1_scRNA-seq_rep1,Donor1_scRNA-seq_rep2,Donor2_scRNA-seq_rep1,Donor2_scRNA-seq_rep2",
    "test_batches": "Donor3_scRNA-seq_rep1,Donor3_scRNA-seq_rep2",
  },
  {
    "dataset_key": "kang_pbmc_gse96583_singlets_raw_counts",
    "name": "train_samples_to_donors_1039_107",
    "split_key": "donor",
    "train_split_key": "sample",
    "test_split_key": "donor",
    "train_batches": "1015_ctrl,1015_stim,1488_ctrl,1488_stim,1256_ctrl,1256_stim,1016_ctrl,1016_stim,1244_ctrl,1244_stim,101_ctrl,101_stim",
    "test_batches": "1039,107",
  },
  {
    "dataset_key": "macaque_retina_gse118480_bipolar_raw_counts",
    "name": "m1m2_to_m3m4",
    "split_key": "macaque_id",
    "train_batches": "M1,M2",
    "test_batches": "M3,M4",
  },
  {
    "dataset_key": "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2",
    "name": "smartseq2_celseq2_to_celseq_fluidigmc1",
    "split_key": "batch",
    "train_batches": "smartseq2,celseq2",
    "test_batches": "celseq,fluidigmc1",
  },
]

REPORT_PRESET_LABELS = {
  "baron_transductive": "Baron transductive comparison (3 repeats)",
  "baron_split_701020": "Baron split 70/10/20 comparison (3 repeats)",
  "common8_methods_harmony": "Common-8 report methods + Harmony complement",
  "loss_transfer_report": "Loss-transfer report protocol (5 seeds)",
  "inductive_report_splits": "Report inductive splits",
}


def _load_report_method_specs() -> Dict[str, Any]:
  """Load canonical report method specs for script generation."""
  try:
    from scrbenchmark.methods import load_method_specs
  except Exception:
    try:
      from methods import load_method_specs
    except Exception:
      return {}

  specs = load_method_specs()
  return {
    spec.name: spec
    for spec in specs.values()
    if getattr(spec, 'report', True)
  }


def _unique_preserve_order(values: List[str]) -> List[str]:
  seen = set()
  out = []
  for value in values:
    if value in seen:
      continue
    seen.add(value)
    out.append(value)
  return out


def _stable_generalist_target_name(raw_path: Any) -> str:
  name = Path(str(raw_path)).name
  aliases = {"pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad"}
  return aliases.get(name, name)


def _load_report_dataset_specs() -> Dict[str, Dict[str, Any]]:
  """Load dataset metadata used by the report presets."""
  if not REPORT_DATASET_TABLE.exists():
    return {}

  try:
    frame = pd.read_csv(REPORT_DATASET_TABLE)
  except Exception:
    return {}

  specs: Dict[str, Dict[str, Any]] = {}
  for _, row in frame.iterrows():
    dataset_key = str(row.get("dataset_key", "")).strip()
    if not dataset_key:
      continue
    label_key = row.get("label_key", "Group")
    batch_key = row.get("dann_batch_column", "batch")
    n_labels = row.get("n_labels", 0)
    data_file = row.get("data_file", f"{dataset_key}.h5ad")
    specs[dataset_key] = {
      "dataset_key": dataset_key,
      "display_name": str(row.get("dataset", dataset_key)).strip() or dataset_key,
      "data_path": str(REPORT_DATA_ROOT / _stable_generalist_target_name(data_file)),
      "label_key": str(label_key if not pd.isna(label_key) else "Group").strip() or "Group",
      "batch_key": str(batch_key if not pd.isna(batch_key) else "batch").strip() or "batch",
      "n_labels": int(n_labels if not pd.isna(n_labels) else 0),
    }
  return specs


def _apply_report_preprocessing_defaults(config: Dict[str, Any]) -> None:
  pre = config.setdefault("preprocessing_params", {})
  pre.update({
    "do_cell_filtering": True,
    "do_gene_filtering": True,
    "min_genes_per_cell": 200,
    "max_genes_per_cell": 10000,
    "min_cells_per_gene": 3,
    "do_normalization": True,
    "do_log_transform": True,
    "target_sum": 20000,
    "do_hvg": True,
    "n_top_genes": 2000,
    "hvg_flavor": "seurat",
    "hvg_strategy": "train_only",
    "do_scaling": True,
    "scale_max_value": 10.0,
    "do_batch_correction": False,
  })


def _apply_dataset_preset(config: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
  specs = _load_report_dataset_specs()
  spec = specs.get(dataset_key, {
    "dataset_key": dataset_key,
    "display_name": dataset_key,
    "data_path": str(REPORT_DATA_ROOT / f"{dataset_key}.h5ad"),
    "label_key": "Group",
    "batch_key": "batch",
    "n_labels": 0,
  })

  config["dataset_key"] = spec["dataset_key"]
  config["uploaded_file_path"] = spec["data_path"]
  config["label_key"] = spec["label_key"]
  config["batch_key"] = spec["batch_key"]
  config["n_labels"] = spec["n_labels"]

  benchmark_setup = config.setdefault("benchmark_setup", {"mode": "benchmark", "original_settings": {}})
  settings = benchmark_setup.setdefault("original_settings", {})
  settings["label_col"] = spec["label_key"]
  settings["batch_col"] = spec["batch_key"]
  return spec


def _set_standard_mode(config: Dict[str, Any]) -> None:
  config["benchmark_configured"] = True
  config["benchmark_setup"] = {"mode": "standard", "original_settings": {}}


def _set_stratified_split_mode(config: Dict[str, Any], *, train_ratio: float, val_ratio: float) -> None:
  label_key = config.get("label_key", "Group")
  batch_key = config.get("batch_key", "batch")
  config["benchmark_configured"] = True
  config["benchmark_setup"] = {
    "mode": "benchmark",
    "original_settings": {
      "mode": "stratified",
      "train_ratio": float(train_ratio),
      "val_ratio": float(val_ratio),
      "test_ratio": max(0.0, 1.0 - float(train_ratio) - float(val_ratio)),
      "use_validation": float(val_ratio) > 0.0,
      "stratify_by_batch": True,
      "stratify_by_labels": True,
      "use_stratified_base": True,
      "batch_col": batch_key,
      "label_col": label_key,
    },
  }


def _report_methods_config(config: Dict[str, Any], methods: List[str]) -> None:
  config["selected_algorithms"] = []
  config["selected_report_methods"] = _unique_preserve_order(methods)
  config["scraw_preset"] = "default"
  config["report_method_n_pcs"] = 50
  config["report_method_harmony_max_iter"] = 10
  config["report_method_harmony_nclust"] = 50


def _manual_protocol_base(config: Dict[str, Any], protocols: List[str], seeds: str) -> Dict[str, Any]:
  protocol_cfg = copy.deepcopy(config.get("manual_protocols", {}))
  protocol_cfg.update({
    "enabled": True,
    "selected_protocols": protocols,
    "seeds": seeds,
    "scib_n_jobs": 4,
    "overwrite": False,
    "verbose": True,
    "extra_params": "",
  })
  config["manual_protocols"] = protocol_cfg
  config["selected_algorithms"] = []
  config["selected_report_methods"] = []
  return protocol_cfg


def _create_report_preset_configs(preset_key: str) -> List[Dict[str, Any]]:
  """Create editable benchmark configs for a report preset."""
  if get_protocol_spec is not None and protocol_to_customize_configs is not None:
    try:
      protocol_spec = get_protocol_spec(preset_key)
      if protocol_spec is not None:
        return protocol_to_customize_configs(protocol_spec)
    except Exception:
      pass

  configs: List[Dict[str, Any]] = []
  all_report_methods = _unique_preserve_order(REPORT_PRIMARY_METHODS + REPORT_HARMONY_METHODS)

  if preset_key == "baron_transductive":
    config = _create_default_config("Rapport - Baron transductive")
    _apply_dataset_preset(config, "baron_human_pancreas")
    _apply_report_preprocessing_defaults(config)
    _set_standard_mode(config)
    config["selected_algorithms"] = list(LOCAL_BARON_COMPARISON_ALGORITHMS)
    config["n_repeats"] = 3
    config["seed"] = 42
    config["output_dir"] = "results/report_design/baron_transductive"
    configs.append(config)

  elif preset_key == "baron_split_701020":
    config = _create_default_config("Rapport - Baron split 70-10-20")
    _apply_dataset_preset(config, "baron_human_pancreas")
    _apply_report_preprocessing_defaults(config)
    _set_stratified_split_mode(config, train_ratio=0.7, val_ratio=0.1)
    config["selected_algorithms"] = list(LOCAL_BARON_COMPARISON_ALGORITHMS)
    config["n_repeats"] = 3
    config["seed"] = 42
    config["output_dir"] = "results/report_design/baron_split_701020"
    configs.append(config)

  elif preset_key == "common8_methods_harmony":
    for dataset_key in COMMON8_DATASET_KEYS:
      config = _create_default_config(f"Rapport - {dataset_key} methods Harmony")
      spec = _apply_dataset_preset(config, dataset_key)
      config["name"] = f"Rapport - {spec['display_name']} methods + Harmony"
      _apply_report_preprocessing_defaults(config)
      _set_standard_mode(config)
      _report_methods_config(config, all_report_methods)
      config["n_repeats"] = 1
      config["seed"] = 42
      config["output_dir"] = "results/report_design/common8_methods_harmony"
      configs.append(config)

  elif preset_key == "loss_transfer_report":
    for dataset_key in LOSS_TRANSFER_DATASET_KEYS:
      config = _create_default_config(f"Rapport - {dataset_key} loss transfer")
      spec = _apply_dataset_preset(config, dataset_key)
      config["name"] = f"Rapport - {spec['display_name']} loss transfer"
      _apply_report_preprocessing_defaults(config)
      _set_standard_mode(config)
      protocol_cfg = _manual_protocol_base(config, ["loss_transfer"], "42-46")
      protocol_cfg["loss_transfer"] = {
        "methods": ["scMAE", "scDeepCluster", "DESC"],
        "variants": ["baseline", "weighted", "density_only", "kmeans", "triplet"],
        "weight_params": "warmup_epochs=55\nrare_triplet_start_epoch=60",
      }
      config["n_repeats"] = 5
      config["output_dir"] = "results/report_design/loss_transfer"
      configs.append(config)

  elif preset_key == "inductive_report_splits":
    for split in INDUCTIVE_SPLIT_PRESETS:
      config = _create_default_config(f"Rapport - {split['dataset_key']} inductive {split['name']}")
      spec = _apply_dataset_preset(config, split["dataset_key"])
      config["name"] = f"Rapport - {spec['display_name']} inductive {split['name']}"
      _apply_report_preprocessing_defaults(config)
      _set_standard_mode(config)
      protocol_cfg = _manual_protocol_base(config, ["inductive"], "42")
      protocol_cfg["inductive"] = {
        "algorithms": list(INDUCTIVE_ALGORITHMS),
        "split_key": split["split_key"],
        "train_split_key": split.get("train_split_key", ""),
        "test_split_key": split.get("test_split_key", ""),
        "train_batches": split["train_batches"],
        "test_batches": split["test_batches"],
        "preset": "default",
        "trial_config_path": "",
        "baseline_runtime_profile": "scrbenchmark-default",
        "skip_existing": True,
      }
      config["n_repeats"] = 1
      config["output_dir"] = "results/report_design/inductive"
      configs.append(config)

  return configs


def _render_report_preset_controls(selected_idx: int) -> None:
  st.markdown("#### Report Presets")
  st.caption("Load editable configurations for the benchmark families used in the report.")
  preset_key = st.selectbox(
    "Preset",
    list(REPORT_PRESET_LABELS.keys()),
    format_func=lambda key: REPORT_PRESET_LABELS[key],
    key="report_preset_selector",
  )

  col_add, col_replace = st.columns(2)
  with col_add:
    if st.button("Add Preset Configs", use_container_width=True):
      new_configs = _create_report_preset_configs(preset_key)
      if not new_configs:
        st.warning("No configuration generated for this preset.")
      else:
        st.session_state.custom_benchmarks.extend(new_configs)
        st.session_state.current_config_idx = len(st.session_state.custom_benchmarks) - len(new_configs)
        st.rerun()
  with col_replace:
    if st.button("Replace Current With Preset", use_container_width=True):
      new_configs = _create_report_preset_configs(preset_key)
      if not new_configs:
        st.warning("No configuration generated for this preset.")
      else:
        st.session_state.custom_benchmarks[selected_idx:selected_idx + 1] = new_configs
        st.session_state.current_config_idx = selected_idx
        st.rerun()

  if preset_key == "inductive_report_splits":
    st.caption(
      "The current inductive runner exposes pca_harmony as the sixth backend; "
      "add a scvi backend before selecting it for native inductive scVI runs."
    )


def _harmony_variant_for(method_name: str, specs: Dict[str, Any]) -> str | None:
  """Return the registered +Harmony method paired with a base report method."""
  if not method_name or method_name == "Harmony" or "+Harmony" in method_name:
    return None
  candidate = f"{method_name}+Harmony"
  return candidate if candidate in specs else None


def _expand_report_methods_with_harmony(config: Dict[str, Any], specs: Dict[str, Any]) -> List[str]:
  """Return selected report methods plus requested post-hoc Harmony variants."""
  selected = [m for m in config.get('selected_report_methods', []) if m]
  expanded = list(selected)
  if not config.get('report_include_harmony_variants', False):
    return expanded

  requested_bases = [
    m for m in config.get('report_harmony_variant_methods', [])
    if m in selected
  ]
  if not requested_bases:
    requested_bases = selected

  for method_name in requested_bases:
    variant = _harmony_variant_for(method_name, specs)
    if variant and variant not in expanded:
      expanded.append(variant)
  return expanded


def _slug(value: Any) -> str:
  text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
  return text.strip('_') or 'method'


def _join_cmd(parts: List[Any]) -> str:
  return " ".join(shlex.quote(str(part)) for part in parts)


def _generate_report_method_commands(config: Dict[str, Any]) -> List[str]:
  """Generate one run_method.py command per selected report method."""
  report_specs = _load_report_method_specs()
  selected_methods = _expand_report_methods_with_harmony(config, report_specs)
  if not selected_methods:
    return []

  pre = config.get('preprocessing_params', {}) or {}
  output_dir = str(config.get('output_dir', 'results') or 'results')
  config_slug = _slug(config.get('name', 'config'))
  data_path = str(config.get('uploaded_file_path', 'data/your_data.h5ad') or 'data/your_data.h5ad')
  dataset_key = str(config.get('dataset_key', '') or Path(data_path).stem)
  label_key = str(config.get('label_key', 'Group') or 'Group')
  batch_key = str(config.get('batch_key', 'batch') or 'batch')
  n_labels = int(config.get('n_labels', 0) or 0)
  base_seed = int(config.get('seed', 42) or 42)
  n_repeats = max(1, int(config.get('n_repeats', 1) or 1))
  device = str(config.get('device_preference', 'auto') or 'auto')
  report_n_pcs = int(config.get('report_method_n_pcs', 50) or 50)
  report_harmony_max_iter = int(config.get('report_method_harmony_max_iter', 10) or 10)
  report_harmony_nclust = int(config.get('report_method_harmony_nclust', 50) or 50)
  report_params = _parse_text_entries(config.get('report_method_params', ''))
  scraw_preset = str(config.get('scraw_preset', 'default') or 'default')
  if scraw_preset not in {'default', 'baron'}:
    scraw_preset = 'default'

  commands = []
  for method_name in selected_methods:
    method_slug = _slug(method_name)
    for repeat_idx in range(n_repeats):
      seed = base_seed + repeat_idx
      output_path = f"{output_dir}/{config_slug}_{method_slug}"
      if n_repeats > 1:
        output_path = f"{output_path}/seed_{seed}"
      cmd = [
        "python3",
        "scripts/reproduction/run_method.py",
        "--method",
        method_name,
        "--data",
        data_path,
        "--output",
        output_path,
        "--dataset-key",
        dataset_key,
        "--label-key",
        label_key,
        "--batch-key",
        batch_key,
        "--n-labels",
        n_labels,
        "--seed",
        seed,
        "--device",
        device,
        "--n-top-genes",
        int(pre.get('n_top_genes', 2000) or 2000),
        "--min-genes-per-cell",
        int(pre.get('min_genes_per_cell', 100) or 100),
        "--max-genes-per-cell",
        int(pre.get('max_genes_per_cell', 10000) or 10000),
        "--min-cells-per-gene",
        int(pre.get('min_cells_per_gene', 3) or 3),
        "--target-sum",
        float(pre.get('target_sum', 20000) or 20000),
        "--scale-max-value",
        float(pre.get('scale_max_value', 10.0) or 10.0),
        "--hvg-flavor",
        str(pre.get('hvg_flavor', 'seurat') or 'seurat'),
        "--n-pcs",
        report_n_pcs,
        "--harmony-max-iter",
        report_harmony_max_iter,
        "--harmony-nclust",
        report_harmony_nclust,
      ]
      if method_name == "scRAW":
        cmd.extend(["--scraw-preset", scraw_preset])
      for param in report_params:
        cmd.extend(["--param", param])
      commands.append(_join_cmd(cmd))

  return commands


def _parse_text_entries(raw: Any) -> List[str]:
  """Parse comma/newline-separated entries while preserving key=value values."""
  if raw is None:
    return []
  entries = []
  for chunk in str(raw).replace('\n', ',').split(','):
    item = chunk.strip()
    if item:
      entries.append(item)
  return entries


def _parse_key_value_text(raw: Any) -> Dict[str, str]:
  out = {}
  for item in _parse_text_entries(raw):
    if '=' not in item:
      continue
    key, value = item.split('=', 1)
    key = key.strip().replace('-', '_')
    if key:
      out[key] = value.strip()
  return out


def _generate_manual_protocol_commands(config: Dict[str, Any]) -> List[str]:
  """Generate commands for user-configured loss-transfer and inductive protocols."""
  settings = config.get('manual_protocols', {}) or {}
  if not settings.get('enabled'):
    return []
  if ManualProtocolConfig is None or build_jobs is None or parse_seeds is None:
    return ["# Manual protocol generation is unavailable: scripts/reproduction/manual_protocols.py could not be imported."]

  selected_protocols = settings.get('selected_protocols', []) or []
  if not selected_protocols:
    return []

  pre = config.get('preprocessing_params', {}) or {}
  output_dir = str(config.get('output_dir', 'results') or 'results')
  config_slug = _slug(config.get('name', 'config'))
  data_path = str(config.get('uploaded_file_path', 'data/your_data.h5ad') or 'data/your_data.h5ad')
  label_key = str(config.get('label_key', 'Group') or 'Group')
  batch_key = str(config.get('batch_key', 'batch') or 'batch')
  n_labels = int(config.get('n_labels', 0) or 0)
  device = str(config.get('device_preference', 'auto') or 'auto')
  seeds_raw = settings.get('seeds', str(config.get('seed', 42)))

  try:
    seeds = tuple(parse_seeds(seeds_raw))
  except Exception:
    seeds = (int(config.get('seed', 42) or 42),)

  common_params = tuple(_parse_text_entries(settings.get('extra_params', '')))
  commands = []

  for protocol in selected_protocols:
    proto = str(protocol).strip()
    loss_cfg = settings.get('loss_transfer', {}) or {}
    harmony_cfg = settings.get('harmony', {}) or {}
    inductive_cfg = settings.get('inductive', {}) or {}

    manual_config = ManualProtocolConfig(
      protocol=proto,
      data=data_path,
      output_root=f"{output_dir}/{config_slug}_manual_protocols",
      dataset_key=str(config.get('dataset_key', '') or Path(data_path).stem),
      label_key=label_key,
      batch_key=batch_key,
      n_labels=n_labels,
      seeds=seeds,
      device=device,
      python_bin="python3",
      scib_n_jobs=int(settings.get('scib_n_jobs', 4) or 4),
      n_top_genes=int(pre.get('n_top_genes', 2000) or 2000),
      min_genes_per_cell=int(pre.get('min_genes_per_cell', 100) or 100),
      max_genes_per_cell=int(pre.get('max_genes_per_cell', 10000) or 10000),
      min_cells_per_gene=int(pre.get('min_cells_per_gene', 3) or 3),
      target_sum=float(pre.get('target_sum', 20000) or 20000),
      scale_max_value=float(pre.get('scale_max_value', 10.0) or 10.0),
      hvg_flavor=str(pre.get('hvg_flavor', 'seurat') or 'seurat'),
      params=common_params,
      overwrite=bool(settings.get('overwrite', False)),
      verbose=bool(settings.get('verbose', True)),
      loss_methods=tuple(loss_cfg.get('methods', ['scMAE', 'scDeepCluster', 'DESC'])),
      loss_variants=tuple(loss_cfg.get('variants', ['baseline', 'weighted'])),
      loss_weight_params=_parse_key_value_text(loss_cfg.get('weight_params', '')),
      harmony_methods=tuple(harmony_cfg.get('methods', REPORT_HARMONY_METHODS)),
      harmony_max_iter=int(harmony_cfg.get('harmony_max_iter', 10) or 10),
      harmony_nclust=int(harmony_cfg.get('harmony_nclust', 50) or 50),
      n_pcs=int(harmony_cfg.get('n_pcs', 50) or 50),
      inductive_algorithms=tuple(inductive_cfg.get('algorithms', ['scraw', 'scname', 'sc_mae', 'scdeepcluster'])),
      split_key=str(inductive_cfg.get('split_key', batch_key) or batch_key),
      train_split_key=str(inductive_cfg.get('train_split_key', '') or ''),
      test_split_key=str(inductive_cfg.get('test_split_key', '') or ''),
      train_batches=tuple(_parse_text_entries(inductive_cfg.get('train_batches', ''))),
      test_batches=tuple(_parse_text_entries(inductive_cfg.get('test_batches', ''))),
      preset=str(inductive_cfg.get('preset', 'default') or 'default'),
      trial_config_path=str(inductive_cfg.get('trial_config_path', '') or ''),
      baseline_runtime_profile=str(inductive_cfg.get('baseline_runtime_profile', 'scrbenchmark-default') or 'scrbenchmark-default'),
      skip_existing=bool(inductive_cfg.get('skip_existing', False)),
    )

    try:
      jobs = build_jobs(manual_config)
      commands.extend([job['command'] for job in jobs])
    except Exception as exc:
      commands.append(f"# Manual protocol '{proto}' is not fully configured: {exc}")

  return commands


def _build_commands_for_config(config: Dict[str, Any]) -> List[str]:
  """Build all CLI commands for one Customize Benchmark configuration."""
  commands_for_config: List[str] = []
  if config.get('selected_algorithms'):
    commands_for_config.append(
      _generate_cli_command(
        compact=False,
        n_repetitions=config.get('n_repeats', 1),
        seed=config.get('seed', 42),
        state_source=config,
      )
    )
  commands_for_config.extend(_generate_report_method_commands(config))
  commands_for_config.extend(_generate_manual_protocol_commands(config))
  return commands_for_config

def render_customize_benchmark_page():
  """Render the customize benchmark page."""
  st.header("Customize Benchmark Scenarios")
  st.caption("Create multiple benchmark configurations and generate execution scripts.")

  # Initialize session state for custom benchmarks
  if 'custom_benchmarks' not in st.session_state:
    st.session_state.custom_benchmarks = []
    # Add a default starting config
    st.session_state.custom_benchmarks.append(_create_default_config("Default Config"))
  
  if 'current_config_idx' not in st.session_state:
    st.session_state.current_config_idx = 0

  # -------------------------------------------------------------------------
  # Sidebar / Top Control: Manage Configurations
  # -------------------------------------------------------------------------
  col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
  
  with col1:
    config_names = [f"{i+1}. {c['name']}" for i, c in enumerate(st.session_state.custom_benchmarks)]
    selected_idx = st.selectbox(
      "Select Configuration to Edit",
      range(len(st.session_state.custom_benchmarks)),
      format_func=lambda i: config_names[i],
      index=st.session_state.current_config_idx
    )
    st.session_state.current_config_idx = selected_idx

  with col2:
    if st.button("New Config", use_container_width=True):
      new_config = _create_default_config(f"Config {len(st.session_state.custom_benchmarks) + 1}")
      st.session_state.custom_benchmarks.append(new_config)
      st.session_state.current_config_idx = len(st.session_state.custom_benchmarks) - 1
      st.rerun()

  with col3:
    if st.button("Duplicate", use_container_width=True):
      current_config = st.session_state.custom_benchmarks[selected_idx]
      # Deep copy to ensure no reference issues
      new_config = copy.deepcopy(current_config)
      new_config['name'] = f"{current_config['name']} (Copy)"
      st.session_state.custom_benchmarks.append(new_config)
      st.session_state.current_config_idx = len(st.session_state.custom_benchmarks) - 1
      st.rerun()

  with col4:
    if st.button("Delete", use_container_width=True):
      if len(st.session_state.custom_benchmarks) > 1:
        st.session_state.custom_benchmarks.pop(selected_idx)
        st.session_state.current_config_idx = max(0, selected_idx - 1)
        st.rerun()
      else:
        st.warning("Cannot delete the last configuration.")

  _render_report_preset_controls(selected_idx)
  if render_protocol_registry_panel is not None:
    render_protocol_registry_panel(selected_idx)

  # -------------------------------------------------------------------------
  # Editor for Selected Configuration
  # -------------------------------------------------------------------------
  st.markdown("---")
  current_config = st.session_state.custom_benchmarks[selected_idx]
  
  # Config Name
  current_config['name'] = st.text_input("Configuration Name", current_config['name'])

  # Tabs for different settings
  tab_data, tab_split, tab_pre, tab_algo, tab_protocols, tab_exec = st.tabs([
    "Data Source",
    "Split Strategy",
    "Preprocessing",
    "Algorithms",
    "Manual Protocols",
    "Execution",
  ])

  with tab_data:
    st.subheader("Data Source")
    current_config['uploaded_file_path'] = st.text_input(
      "Data File Path (H5AD)", 
      current_config.get('uploaded_file_path', 'data/dataset.h5ad'),
      help="Path to the .h5ad file relative to the project root."
    )
    default_dataset_key = current_config.get('dataset_key') or Path(
      str(current_config.get('uploaded_file_path', 'dataset.h5ad'))
    ).stem
    c_data1, c_data2, c_data3, c_data4 = st.columns(4)
    with c_data1:
      current_config['dataset_key'] = st.text_input(
        "Dataset Key",
        str(default_dataset_key),
        help="Canonical dataset identifier used in reproduction commands and output naming.",
        key=f"dataset_key_{selected_idx}"
      )
    with c_data2:
      current_config['label_key'] = st.text_input(
        "Label Column",
        current_config.get('label_key', 'Group'),
        help="Ground-truth cell type column used for metrics and reproduction runners.",
        key=f"label_key_{selected_idx}"
      )
    with c_data3:
      current_config['batch_key'] = st.text_input(
        "Batch Column",
        current_config.get('batch_key', 'batch'),
        help="Batch/sample column used for batch metrics and batch-aware methods.",
        key=f"batch_key_data_{selected_idx}"
      )
    with c_data4:
      current_config['n_labels'] = st.number_input(
        "Expected Classes",
        min_value=0,
        max_value=500,
        value=int(current_config.get('n_labels', 0) or 0),
        help="Required by report reproduction methods; use 0 only while drafting commands.",
        key=f"n_labels_{selected_idx}"
      )

  with tab_split:
    st.subheader("Benchmark Split Strategy")
    
    # Setup structure
    if 'benchmark_setup' not in current_config:
      current_config['benchmark_setup'] = {'mode': 'benchmark', 'original_settings': {}}
      current_config['benchmark_configured'] = True

    # Number of repeats (Global setting for this config)
    st.caption("General Execution Settings")
    n_repeats = st.number_input(
      "Number of Repetitions", 
      min_value=1, 
      max_value=100, 
      value=current_config.get('n_repeats', 1),
      help="Number of times to run each algorithm (with different seeds if not fixed).",
      key=f"n_repeats_{selected_idx}"
    )
    current_config['n_repeats'] = n_repeats
    st.divider()

    settings = current_config['benchmark_setup']['original_settings']
    
    # Check if it was standard mode
    mode_idx = 0
    if current_config['benchmark_setup'].get('mode') == 'standard':
      mode_idx = 2
    elif settings.get('mode') == 'batch':
      mode_idx = 1
    
    split_mode = st.radio(
      "Split Mode",
      ["Split dataset (Classical)", "Split dataset (By Batch)", "All dataset"],
      index=mode_idx,
      key=f"split_mode_{selected_idx}"
    )
    
    if split_mode == "All dataset":
      current_config['benchmark_setup']['mode'] = 'standard'
      st.info("Uses the entire dataset for clustering. Best for exploration.")

      # Batch Balancing Options for Standard Mode
      st.markdown("##### Batch Balancing (Optional)")
      st.caption("Subsample larger batches so each contributes equally to the analysis.")

      # Initialize balance_settings_standard if not present
      if 'balance_settings_standard' not in current_config:
        current_config['balance_settings_standard'] = {}

      balance_std = current_config.get('balance_settings_standard', {})

      enable_balance_std = st.checkbox(
        "Enable Batch Balancing",
        value=bool(balance_std),
        key=f"enable_balance_std_{selected_idx}",
        help="Reduce larger batches to equalize contributions from each batch."
      )

      if enable_balance_std:
        c1, c2 = st.columns(2)
        with c1:
          balance_std['batch_col'] = st.text_input(
            "Batch Column",
            value=balance_std.get('batch_col', 'batch'),
            key=f"balance_std_col_{selected_idx}",
            help="Column containing batch identifiers"
          )
        with c2:
          target_raw = st.text_input(
            "Target Cells per Batch",
            value=str(balance_std.get('target', 'auto')),
            key=f"balance_std_target_{selected_idx}",
            help='Use "auto" (default) to pick the smallest batch size, or set a positive integer.'
          )
          parsed_target = _parse_balance_target(target_raw)
          if parsed_target is None:
            st.warning('Invalid balance target. Falling back to "auto".')
            parsed_target = 'auto'
          balance_std['target'] = parsed_target
        current_config['balance_settings_standard'] = balance_std
      else:
        current_config['balance_settings_standard'] = {}

    elif split_mode == "Split dataset (Classical)":
      current_config['benchmark_setup']['mode'] = 'benchmark'
      settings['mode'] = 'stratified'
      
      c1, c2, c3 = st.columns(3)
      train_ratio = c1.number_input("Train Ratio", 0.1, 0.9, settings.get('train_ratio', 0.8), key=f"train_ratio_{selected_idx}")
      val_ratio = c2.number_input("Val Ratio", 0.0, 0.5, settings.get('val_ratio', 0.1), key=f"val_ratio_{selected_idx}")
      test_ratio = 1.0 - train_ratio - val_ratio
      c3.metric("Test Ratio", f"{test_ratio:.2f}")
      
      settings['train_ratio'] = train_ratio
      settings['val_ratio'] = val_ratio
      settings['test_ratio'] = test_ratio

      # Balancing options (Requested by user to match software implementation)
      st.markdown("##### Balancing Options")
      settings['balance'] = st.checkbox("Balance Batches", settings.get('balance', False), key=f"balance_classic_{selected_idx}")
      if settings['balance']:
        target_raw = st.text_input(
          "Balance Target (cells/batch)",
          value=str(settings.get('balance_target', 'auto')),
          key=f"bal_target_classic_{selected_idx}",
          help='Use "auto" (default) to pick the smallest batch size, or set a positive integer.'
        )
        parsed_target = _parse_balance_target(target_raw)
        if parsed_target is None:
          st.warning('Invalid balance target. Falling back to "auto".')
          parsed_target = 'auto'
        settings['balance_target'] = parsed_target
        balance_mode_options = ['eliminate', 'reinject_train', 'reinject_both']
        balance_mode_labels = {
          'eliminate': 'Eliminate excess cells (standard)',
          'reinject_train': 'Reinject into TRAIN (no data loss)',
          'reinject_both': 'Reinject into TRAIN + VAL (no data loss)'
        }
        settings['balance_mode'] = st.selectbox(
          "Balance Mode",
          balance_mode_options,
          index=balance_mode_options.index(settings.get('balance_mode', 'eliminate')),
          format_func=lambda x: balance_mode_labels[x],
          key=f"bal_mode_classic_{selected_idx}",
          help="How to handle excess cells during balancing"
        )

      st.markdown("##### Stratification Options")

      settings['stratify_by_batch'] = st.checkbox("Stratify by Batch", settings.get('stratify_by_batch', True), key=f"strat_batch_{selected_idx}")
      settings['stratify_by_labels'] = st.checkbox("Stratify by Labels", settings.get('stratify_by_labels', True), key=f"strat_labels_{selected_idx}")
      
      if settings['stratify_by_batch']:
        settings['batch_col'] = st.text_input("Batch Column", settings.get('batch_col', 'batch'), key=f"batch_col_{selected_idx}")

    else:
      current_config['benchmark_setup']['mode'] = 'benchmark'
      settings['mode'] = 'batch'
      st.info("Train on selected batches. Choose whether test is ALL batches or explicit unseen batches.")
      settings['batch_col'] = st.text_input("Batch Column for Split", settings.get('batch_col', 'batch'), key=f"batch_col_b_{selected_idx}")
      train_batches_str = st.text_input("Train Batches (comma separated)", ",".join(settings.get('train_batches', [])), key=f"train_batches_{selected_idx}")
      settings['train_batches'] = [b.strip() for b in train_batches_str.split(',') if b.strip()]
      if not settings['train_batches']:
        st.warning("Select at least one training batch.")

      st.markdown("##### Validation & Ratios")
      settings['use_validation'] = st.checkbox(
        "Create validation set",
        value=bool(settings.get('use_validation', True)),
        key=f"use_validation_b_{selected_idx}",
        help="Validation is used for LR optimization / held-out metrics."
      )
      if settings['use_validation']:
        settings['val_ratio'] = st.number_input(
          "Validation Ratio",
          min_value=0.0,
          max_value=0.5,
          value=float(settings.get('val_ratio', 0.1)),
          step=0.05,
          key=f"val_ratio_b_{selected_idx}"
        )
      else:
        settings['val_ratio'] = 0.0

      train_ratio_b = st.number_input(
        "Train Ratio (for ALL-batches test mode)",
        min_value=0.5,
        max_value=0.9,
        value=float(settings.get('train_ratio', 0.7)),
        step=0.05,
        key=f"train_ratio_b_{selected_idx}"
      )
      test_ratio_b = 1.0 - train_ratio_b - float(settings.get('val_ratio', 0.0))
      st.metric("Test Ratio (ALL-batches mode)", f"{test_ratio_b:.2f}")
      if test_ratio_b <= 0:
        st.error("Train + Val must be < 1.0 for ALL-batches test mode.")
      settings['train_ratio'] = float(train_ratio_b)
      settings['test_ratio'] = float(max(0.0, test_ratio_b))

      st.markdown("##### Test Strategy")
      test_strategy = st.radio(
        "Evaluate generalization on:",
        options=['all', 'explicit'],
        format_func=lambda x: {
          'all': 'All Batches (stratified base)',
          'explicit': 'Explicit Test Batches (strict unseen/generalization)'
        }[x],
        index=0 if settings.get('use_stratified_base', True) else 1,
        key=f"test_strategy_b_{selected_idx}"
      )

      if test_strategy == 'all':
        settings['use_stratified_base'] = True
        settings['test_batches'] = None
        st.info(
          "TEST is a stratified split over all batches. "
          "TRAIN/VAL are filtered to selected train batches."
        )
      else:
        settings['use_stratified_base'] = False
        test_batches_str = st.text_input(
          "Test Batches (comma separated)",
          ",".join(settings.get('test_batches', []) or []),
          key=f"test_batches_{selected_idx}",
          help="Batches reserved for strict unseen-batch evaluation."
        )
        settings['test_batches'] = [b.strip() for b in test_batches_str.split(',') if b.strip()]
        if not settings['test_batches']:
          st.warning("Specify at least one test batch in strict mode.")
        st.caption(
          "Strict mode: train uses selected train batches, test uses selected test batches. "
          "Train/Test ratios are informational here; val_ratio controls only validation carve-out from train."
        )
        if settings['use_validation']:
          st.caption("Set 'Create validation set' OFF for 100% train-batch usage.")
      
      # Additional settings like balance
      settings['balance'] = st.checkbox("Balance Batches", settings.get('balance', False), key=f"balance_{selected_idx}")
      if settings['balance']:
        target_raw = st.text_input(
          "Balance Target (cells/batch)",
          value=str(settings.get('balance_target', 'auto')),
          key=f"bal_target_{selected_idx}",
          help='Use "auto" (default) to pick the smallest batch size, or set a positive integer.'
        )
        parsed_target = _parse_balance_target(target_raw)
        if parsed_target is None:
          st.warning('Invalid balance target. Falling back to "auto".')
          parsed_target = 'auto'
        settings['balance_target'] = parsed_target
        balance_mode_options = ['eliminate', 'reinject_train', 'reinject_both']
        balance_mode_labels = {
          'eliminate': 'Eliminate excess cells (standard)',
          'reinject_train': 'Reinject into TRAIN (no data loss)',
          'reinject_both': 'Reinject into TRAIN + VAL (no data loss)'
        }
        settings['balance_mode'] = st.selectbox(
          "Balance Mode",
          balance_mode_options,
          index=balance_mode_options.index(settings.get('balance_mode', 'eliminate')),
          format_func=lambda x: balance_mode_labels[x],
          key=f"bal_mode_{selected_idx}",
          help="How to handle excess cells during balancing"
        )

      # Stratification options for batch split
      st.markdown("##### Stratification Options")
      col_strat1, col_strat2 = st.columns(2)
      with col_strat1:
        settings['stratify_by_labels'] = st.checkbox(
          "Stratify by Labels", 
          value=settings.get('stratify_by_labels', True),
          help="Preserve cell type balance within batches.",
          key=f"strat_labels_b_{selected_idx}"
        )
      with col_strat2:
        st.caption(
          "Base stratification is controlled by Test Strategy above."
        )

  with tab_pre:
    st.subheader("Preprocessing Pipeline")
    pre_params = current_config.get('preprocessing_params', {})

    # ===== Section 1: Quality Filtering =====
    st.markdown("#### 1. Quality Filtering")
    st.caption("Remove low-quality cells and lowly expressed genes")

    c1, c2 = st.columns(2)
    with c1:
      pre_params['do_cell_filtering'] = st.checkbox(
        "Cell Filtering",
        pre_params.get('do_cell_filtering', True),
        key=f"do_cell_filt_{selected_idx}",
        help="Remove cells with too few or too many genes"
      )
    with c2:
      pre_params['do_gene_filtering'] = st.checkbox(
        "Gene Filtering",
        pre_params.get('do_gene_filtering', True),
        key=f"do_gene_filt_{selected_idx}",
        help="Remove lowly expressed genes"
      )

    if pre_params.get('do_cell_filtering', True):
      c1, c2 = st.columns(2)
      with c1:
        pre_params['min_genes_per_cell'] = st.number_input(
          "Min Genes/Cell", 0, 2000,
          pre_params.get('min_genes_per_cell', 100),
          key=f"min_genes_{selected_idx}"
        )
      with c2:
        pre_params['max_genes_per_cell'] = st.number_input(
          "Max Genes/Cell", 1000, 50000,
          pre_params.get('max_genes_per_cell', 10000),
          key=f"max_genes_{selected_idx}"
        )

    if pre_params.get('do_gene_filtering', True):
      pre_params['min_cells_per_gene'] = st.number_input(
        "Min Cells/Gene", 1, 50,
        pre_params.get('min_cells_per_gene', 3),
        key=f"min_cells_{selected_idx}"
      )

    # ===== Section 2: Normalization & Transformation =====
    st.markdown("#### 2. Normalization & Transformation")
    st.caption("Library size normalization and log transformation")

    c1, c2 = st.columns(2)
    with c1:
      pre_params['do_normalization'] = st.checkbox(
        "Normalization",
        pre_params.get('do_normalization', True),
        key=f"do_norm_{selected_idx}",
        help="Normalize each cell to a fixed total count"
      )
    with c2:
      pre_params['do_log_transform'] = st.checkbox(
        "Log Transform",
        pre_params.get('do_log_transform', True),
        key=f"do_log_{selected_idx}",
        help="Apply log1p transformation"
      )

    if pre_params.get('do_normalization', True):
      pre_params['target_sum'] = st.number_input(
        "Target Sum", 1000, 1000000,
        pre_params.get('target_sum', 20000),
        step=1000,
        key=f"target_sum_{selected_idx}",
        help="Normalization target sum (typically 20000 for this setup)"
      )

    # ===== Section 3: Feature Selection (HVG) =====
    st.markdown("#### 3. Feature Selection")

    pre_params['do_hvg'] = st.checkbox(
      "HVG Selection",
      pre_params.get('do_hvg', True),
      key=f"do_hvg_{selected_idx}",
      help="Select highly variable genes"
    )

    if pre_params.get('do_hvg', True):
      c1, c2 = st.columns(2)
      with c1:
        pre_params['n_top_genes'] = st.number_input(
          "N Top Genes", 500, 10000,
          pre_params.get('n_top_genes', 2000),
          key=f"n_hvg_{selected_idx}"
        )
      with c2:
        hvg_flavors = ['seurat', 'seurat_v3', 'cell_ranger']
        pre_params['hvg_flavor'] = st.selectbox(
          "HVG Method",
          hvg_flavors,
          index=hvg_flavors.index(pre_params.get('hvg_flavor', 'seurat')),
          key=f"hvg_flavor_{selected_idx}"
        )

      # HVG Strategy (for benchmark mode)
      if current_config.get('benchmark_setup', {}).get('mode') == 'benchmark':
        hvg_strategies = ['train_only', 'per_batch_union']
        strategy_labels = {
          'train_only': 'Train Only (No Leakage)',
          'per_batch_union': 'Per-Batch Union'
        }
        pre_params['hvg_strategy'] = st.selectbox(
          "HVG Strategy",
          hvg_strategies,
          format_func=lambda x: strategy_labels[x],
          index=hvg_strategies.index(pre_params.get('hvg_strategy', 'train_only')),
          key=f"hvg_strategy_{selected_idx}",
          help="How to select HVGs in benchmark mode"
        )

    # ===== Section 4: Scaling =====
    st.markdown("#### 4. Scaling")

    pre_params['do_scaling'] = st.checkbox(
      "Scaling",
      pre_params.get('do_scaling', True),
      key=f"do_scale_{selected_idx}",
      help="Z-score standardization"
    )

    if pre_params.get('do_scaling', True):
      pre_params['scale_max_value'] = st.number_input(
        "Scale Max Value", 1.0, 50.0,
        pre_params.get('scale_max_value', 10.0),
        step=1.0,
        key=f"scale_max_{selected_idx}",
        help="Clip values to this maximum after scaling"
      )

    # ===== Section 5: Batch Correction (optional) =====
    st.markdown("#### 5. Batch Correction (Optional)")

    pre_params['do_batch_correction'] = st.checkbox(
      "Batch Correction",
      pre_params.get('do_batch_correction', False),
      key=f"do_bc_{selected_idx}",
      help="Apply batch correction to remove technical variation"
    )

    if pre_params.get('do_batch_correction', False):
      c1, c2 = st.columns(2)
      with c1:
        bc_methods = ['scvi', 'sysvi']
        default_method = pre_params.get('batch_correction_method', 'scvi')
        if default_method not in bc_methods:
          default_method = 'scvi'
        pre_params['batch_correction_method'] = st.selectbox(
          "Batch Correction Method",
          bc_methods,
          index=bc_methods.index(default_method),
          key=f"bc_method_{selected_idx}"
        )
      with c2:
        pre_params['batch_correction_batch_key'] = st.text_input(
          "Batch Key",
          pre_params.get('batch_correction_batch_key', 'batch'),
          key=f"bc_key_{selected_idx}"
        )

    current_config['preprocessing_params'] = pre_params

  with tab_algo:
    st.subheader("Algorithm Selection")
    registry_algos = list(AlgorithmRegistry.get_all().keys())
    available_algos = sorted(
      registry_algos,
      key=lambda name: AlgorithmRegistry.get(name).get_info().display_name.lower()
      if AlgorithmRegistry.get(name) is not None else name.lower()
    )
    
    # Determine multiselect default
    valid_defaults = [a for a in current_config.get('selected_algorithms', []) if a in available_algos]
    
    selected = st.multiselect(
      "Select Algorithms to Run",
      available_algos,
      default=valid_defaults,
      format_func=lambda x: AlgorithmRegistry.get(x).get_info().display_name,
      key=f"algo_sel_{selected_idx}"
    )
    current_config['selected_algorithms'] = selected
    
    # Display individual parameter editors for each selected algorithm
    if selected:
      st.markdown("---")
      st.subheader("Individual Algorithm Hyperparameters")
      st.caption("Each selected algorithm has its own independent parameter set in this configuration.")
      
      if 'algorithm_params' not in current_config:
        current_config['algorithm_params'] = {}

      algo_tabs = st.tabs([AlgorithmRegistry.get(name).get_info().display_name for name in selected])
      for tab, algo_name in zip(algo_tabs, selected):
        with tab:
          _render_custom_algo_params(algo_name, current_config, selected_idx)

      summary_df = _build_algo_modifications_summary(current_config, selected)
      if not summary_df.empty:
        st.markdown("##### Modified Parameters Summary")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Report Methods and Harmony Variants")
    report_specs = _load_report_method_specs()
    if report_specs:
      available_methods = sorted(
        report_specs.keys(),
        key=lambda name: report_specs[name].display_name.lower()
      )
      valid_method_defaults = [
        m for m in current_config.get('selected_report_methods', [])
        if m in available_methods
      ]
      selected_report_methods = st.multiselect(
        "Select report methods to rerun",
        available_methods,
        default=valid_method_defaults,
        format_func=lambda name: f"{report_specs[name].display_name} ({report_specs[name].runner_kind})",
        key=f"report_method_sel_{selected_idx}",
        help="These commands use scripts/reproduction/run_method.py and cover the report method registry."
      )
      current_config['selected_report_methods'] = selected_report_methods
      if "scRAW" in selected_report_methods:
        current_preset = str(current_config.get('scraw_preset', 'default') or 'default')
        if current_preset not in ['default', 'baron']:
          current_preset = 'default'
        current_config['scraw_preset'] = st.selectbox(
          "scRAW preset",
          ['default', 'baron'],
          index=['default', 'baron'].index(current_preset),
          key=f"scraw_report_preset_{selected_idx}",
          help="default uses the vendored 0017 configuration; baron uses the vendored Baron configuration.",
        )
      else:
        current_config['scraw_preset'] = str(current_config.get('scraw_preset', 'default') or 'default')

      compatible_harmony_bases = [
        method for method in selected_report_methods
        if _harmony_variant_for(method, report_specs)
      ]
      include_harmony = st.checkbox(
        "Also evaluate + Harmony variants when available",
        value=bool(current_config.get('report_include_harmony_variants', False)),
        key=f"report_include_harmony_{selected_idx}",
        help=(
          "Adds matched methods such as scMAE+Harmony or scNAME+Harmony to the "
          "generated commands. For deep learning methods, Harmony is applied "
          "after training on the learned latent space, then final clustering is rerun."
        ),
      )
      current_config['report_include_harmony_variants'] = include_harmony
      if include_harmony and compatible_harmony_bases:
        default_harmony_bases = [
          method for method in current_config.get('report_harmony_variant_methods', compatible_harmony_bases)
          if method in compatible_harmony_bases
        ]
        if not default_harmony_bases:
          default_harmony_bases = compatible_harmony_bases
        current_config['report_harmony_variant_methods'] = st.multiselect(
          "Apply +Harmony to these selected methods",
          compatible_harmony_bases,
          default=default_harmony_bases,
          format_func=lambda name: report_specs[name].display_name,
          key=f"report_harmony_variant_methods_{selected_idx}",
          help="This is an inclusion list: only checked base methods get an additional +Harmony run.",
        )
        added = [
          _harmony_variant_for(method, report_specs)
          for method in current_config['report_harmony_variant_methods']
        ]
        added = [method for method in added if method]
        if added:
          st.caption("Generated in addition: " + ", ".join(added))
      elif include_harmony:
        current_config['report_harmony_variant_methods'] = []
        st.info("No selected report method has a registered +Harmony variant.")
      else:
        current_config['report_harmony_variant_methods'] = []

      st.caption(
        "Preprocessing batch correction is applied before model fitting. "
        "+Harmony variants are different: for deep learning methods, Harmony "
        "is applied post-hoc on the latent space learned at the end of training."
      )
      current_config['report_method_params'] = st.text_area(
        "Extra --param entries for report methods",
        value=str(current_config.get('report_method_params', '')),
        placeholder="my_method:lr=0.001\nmy_method:batch_size=128",
        key=f"report_method_params_{selected_idx}",
        help="One entry per line or comma. These are appended to generated run_method.py commands."
      )
      uses_harmony_params = include_harmony or any(
        method == "Harmony" or "+Harmony" in method
        for method in selected_report_methods
      )
      if uses_harmony_params:
        c_pca, c_hmax, c_hclust = st.columns(3)
        current_config['report_method_n_pcs'] = c_pca.number_input(
          "Report method PCA dimensions",
          min_value=1,
          max_value=512,
          value=int(current_config.get('report_method_n_pcs', 50) or 50),
          key=f"report_method_n_pcs_{selected_idx}",
        )
        current_config['report_method_harmony_max_iter'] = c_hmax.number_input(
          "Harmony max iterations",
          min_value=1,
          max_value=100,
          value=int(current_config.get('report_method_harmony_max_iter', 10) or 10),
          key=f"report_method_harmony_max_iter_{selected_idx}",
        )
        current_config['report_method_harmony_nclust'] = c_hclust.number_input(
          "Harmony clusters",
          min_value=1,
          max_value=1000,
          value=int(current_config.get('report_method_harmony_nclust', 50) or 50),
          key=f"report_method_harmony_nclust_{selected_idx}",
        )
      else:
        current_config['report_method_n_pcs'] = st.number_input(
          "Report method PCA dimensions",
          min_value=1,
          max_value=512,
          value=int(current_config.get('report_method_n_pcs', 50) or 50),
          key=f"report_method_n_pcs_{selected_idx}",
          help="Used by report-method runners that need a PCA representation.",
        )
        current_config['report_method_harmony_max_iter'] = int(
          current_config.get('report_method_harmony_max_iter', 10) or 10
        )
        current_config['report_method_harmony_nclust'] = int(
          current_config.get('report_method_harmony_nclust', 50) or 50
        )
      if selected_report_methods and int(current_config.get('n_labels', 0) or 0) <= 0:
        st.warning("Expected Classes must be set above 0 before executing generated report-method commands.")
    else:
      current_config['selected_report_methods'] = []
      st.info("No report method registry was found.")
    
    # Simple Global Architecture
    st.markdown("---")
    st.markdown("#### Global Autoencoder Architecture Override")
    global_net = current_config.get('global_network_config', {'enabled': False})
    use_global = st.checkbox("Override Architecture for All Models", global_net.get('enabled', False), key=f"use_global_{selected_idx}")
    global_net['enabled'] = use_global
    
    if use_global:
      c1, c2 = st.columns(2)
      global_net['encoder_layers'] = c1.text_input("Encoder Layers", global_net.get('encoder_layers', '512,256,64'), key=f"enc_l_{selected_idx}")
      global_net['z_dim'] = c2.number_input("Latent Dim", 0, 512, global_net.get('z_dim', 32), key=f"z_dim_{selected_idx}")
      st.caption("Decoder will be symmetric.")
      
    current_config['global_network_config'] = global_net

  with tab_protocols:
    st.subheader("Manual Protocols")
    protocol_cfg = current_config.get('manual_protocols', {}) or {}

    protocol_cfg['enabled'] = st.checkbox(
      "Generate manual protocol commands",
      value=bool(protocol_cfg.get('enabled', False)),
      key=f"manual_protocols_enabled_{selected_idx}",
    )

    protocol_options = ['loss_transfer', 'inductive']
    selected_protocols = st.multiselect(
      "Protocols",
      protocol_options,
      default=[p for p in protocol_cfg.get('selected_protocols', []) if p in protocol_options],
      key=f"manual_protocols_selected_{selected_idx}",
      help=(
        "Harmony is configured in the Algorithms tab as '+Harmony variants'. "
        "Manual protocols are kept for loss-transfer and custom inductive splits."
      ),
    )
    protocol_cfg['selected_protocols'] = selected_protocols

    c_proto1, c_proto2, c_proto3 = st.columns(3)
    with c_proto1:
      protocol_cfg['seeds'] = st.text_input(
        "Seeds",
        value=str(protocol_cfg.get('seeds', current_config.get('seed', 42))),
        key=f"manual_protocols_seeds_{selected_idx}",
      )
    with c_proto2:
      protocol_cfg['scib_n_jobs'] = st.number_input(
        "scIB jobs",
        min_value=1,
        max_value=64,
        value=int(protocol_cfg.get('scib_n_jobs', 4)),
        key=f"manual_protocols_scib_jobs_{selected_idx}",
      )
    with c_proto3:
      protocol_cfg['overwrite'] = st.checkbox(
        "Overwrite",
        value=bool(protocol_cfg.get('overwrite', False)),
        key=f"manual_protocols_overwrite_{selected_idx}",
      )
    protocol_cfg['verbose'] = st.checkbox(
      "Verbose logs",
      value=bool(protocol_cfg.get('verbose', True)),
      key=f"manual_protocols_verbose_{selected_idx}",
    )
    protocol_cfg['extra_params'] = st.text_area(
      "Extra --param entries",
      value=str(protocol_cfg.get('extra_params', '')),
      height=88,
      key=f"manual_protocols_extra_params_{selected_idx}",
      placeholder="sc_mae:epochs=80\nscdeepcluster_scraw_weighted:rare_triplet_weight=0.05",
    )

    if 'loss_transfer' in selected_protocols:
      st.markdown("#### Loss Transfer")
      loss_cfg = protocol_cfg.get('loss_transfer', {}) or {}
      loss_methods = ['scMAE', 'scDeepCluster', 'DESC']
      loss_variants = ['baseline', 'weighted', 'density_only', 'kmeans', 'triplet']
      loss_cfg['methods'] = st.multiselect(
        "Loss-transfer methods",
        loss_methods,
        default=[m for m in loss_cfg.get('methods', loss_methods) if m in loss_methods],
        key=f"loss_methods_{selected_idx}",
      )
      loss_cfg['variants'] = st.multiselect(
        "Loss-transfer variants",
        loss_variants,
        default=[v for v in loss_cfg.get('variants', ['baseline', 'weighted']) if v in loss_variants],
        key=f"loss_variants_{selected_idx}",
      )
      loss_cfg['weight_params'] = st.text_area(
        "Weighted-loss parameter overrides",
        value=str(loss_cfg.get('weight_params', '')),
        height=88,
        key=f"loss_weight_params_{selected_idx}",
        placeholder="warmup_epochs=55\npseudo_label_method=kmeans\nrare_triplet_weight=0.05",
      )
      protocol_cfg['loss_transfer'] = loss_cfg

    if 'inductive' in selected_protocols:
      st.markdown("#### Inductive")
      inductive_cfg = protocol_cfg.get('inductive', {}) or {}
      inductive_algorithms = ['scraw', 'scname', 'sc_mae', 'scdeepcluster', 'scaide', 'pca_harmony']
      inductive_cfg['algorithms'] = st.multiselect(
        "Inductive algorithms",
        inductive_algorithms,
        default=[a for a in inductive_cfg.get('algorithms', ['scraw', 'scname', 'sc_mae', 'scdeepcluster']) if a in inductive_algorithms],
        key=f"inductive_algorithms_{selected_idx}",
      )
      c_i1, c_i2, c_i3 = st.columns(3)
      with c_i1:
        inductive_cfg['split_key'] = st.text_input(
          "Default split key",
          value=str(inductive_cfg.get('split_key', current_config.get('batch_key', 'batch'))),
          key=f"inductive_split_key_{selected_idx}",
        )
      with c_i2:
        inductive_cfg['train_split_key'] = st.text_input(
          "Train split key",
          value=str(inductive_cfg.get('train_split_key', '')),
          key=f"inductive_train_split_key_{selected_idx}",
        )
      with c_i3:
        inductive_cfg['test_split_key'] = st.text_input(
          "Test split key",
          value=str(inductive_cfg.get('test_split_key', '')),
          key=f"inductive_test_split_key_{selected_idx}",
        )
      inductive_cfg['train_batches'] = st.text_input(
        "Train groups",
        value=str(inductive_cfg.get('train_batches', '')),
        key=f"inductive_train_batches_{selected_idx}",
        placeholder="human1,human2,human3",
      )
      inductive_cfg['test_batches'] = st.text_input(
        "Test groups",
        value=str(inductive_cfg.get('test_batches', '')),
        key=f"inductive_test_batches_{selected_idx}",
        placeholder="human4",
      )
      c_i4, c_i5 = st.columns(2)
      with c_i4:
        inductive_cfg['preset'] = st.selectbox(
          "scRAW preset",
          ['default', 'baron'],
          index=['default', 'baron'].index(inductive_cfg.get('preset', 'default'))
          if inductive_cfg.get('preset', 'default') in ['default', 'baron'] else 0,
          key=f"inductive_preset_{selected_idx}",
        )
      with c_i5:
        inductive_cfg['baseline_runtime_profile'] = st.selectbox(
          "Runtime profile",
          ['scrbenchmark-default', 'debug-fast'],
          index=['scrbenchmark-default', 'debug-fast'].index(inductive_cfg.get('baseline_runtime_profile', 'scrbenchmark-default'))
          if inductive_cfg.get('baseline_runtime_profile', 'scrbenchmark-default') in ['scrbenchmark-default', 'debug-fast'] else 0,
          key=f"inductive_runtime_profile_{selected_idx}",
        )
      inductive_cfg['trial_config_path'] = st.text_input(
        "Trial config path",
        value=str(inductive_cfg.get('trial_config_path', '')),
        key=f"inductive_trial_config_path_{selected_idx}",
      )
      inductive_cfg['skip_existing'] = st.checkbox(
        "Skip existing",
        value=bool(inductive_cfg.get('skip_existing', False)),
        key=f"inductive_skip_existing_{selected_idx}",
      )
      protocol_cfg['inductive'] = inductive_cfg

    current_config['manual_protocols'] = protocol_cfg

  with tab_exec:
    st.subheader("Execution Settings")

    # Random Seed
    st.markdown("#### Reproducibility")
    c1, c2 = st.columns(2)
    with c1:
      seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=current_config.get('seed', 42),
        key=f"seed_{selected_idx}",
        help="Base random seed for reproducibility. Each repetition uses seed + run_id."
      )
      current_config['seed'] = seed

    with c2:
      # n_repeats is already in tab_split, but we can show it here too for clarity
      st.metric("Repetitions", current_config.get('n_repeats', 1))
      st.caption("(Set in Split Strategy tab)")

    # Device Selection
    st.markdown("#### Compute Device")
    device_options = ['auto', 'cpu', 'cuda', 'mps']
    device_labels = {
      'auto': 'Auto (detect best available)',
      'cpu': 'CPU only',
      'cuda': 'NVIDIA GPU (CUDA)',
      'mps': 'Apple Silicon GPU (MPS)'
    }
    current_device = current_config.get('device_preference', 'auto')
    if current_device not in device_options:
      current_device = 'auto'

    device = st.selectbox(
      "Compute Device",
      device_options,
      index=device_options.index(current_device),
      format_func=lambda x: device_labels[x],
      key=f"device_{selected_idx}",
      help="Select the device to run algorithms on."
    )
    current_config['device_preference'] = device

    # Output Configuration
    st.markdown("#### Output")
    output_dir = st.text_input(
      "Output Directory",
      value=current_config.get('output_dir', 'results'),
      key=f"output_dir_{selected_idx}",
      help="Directory where results will be saved. Relative to project root."
    )
    current_config['output_dir'] = output_dir

    c1, c2 = st.columns(2)
    with c1:
      save_labels = st.checkbox(
        "Save Predicted Labels",
        value=current_config.get('save_labels', True),
        key=f"save_labels_{selected_idx}",
        help="Save predicted cluster labels to CSV files."
      )
      current_config['save_labels'] = save_labels

    with c2:
      save_embeddings = st.checkbox(
        "Save Embeddings",
        value=current_config.get('save_embeddings', False),
        key=f"save_embeddings_{selected_idx}",
        help="Save latent embeddings (can be large files)."
      )
      current_config['save_embeddings'] = save_embeddings

    st.markdown("#### Validation-Driven LR Optimization")
    st.caption("Optimize learning rate(s) on validation split before final benchmark fit.")
    val_opt = current_config.get('validation_optimization', {})

    enable_lr_opt = st.checkbox(
      "Enable LR Optimization on Validation",
      value=val_opt.get('enable_lr_optimization', True),
      key=f"enable_lr_opt_{selected_idx}",
      help="Applies to all selected algorithms that expose LR-like hyperparameters (lr, *_lr, learning_rate)."
    )

    col_lr1, col_lr2 = st.columns(2)
    with col_lr1:
      lr_metric = st.selectbox(
        "Validation Metric",
        options=['NMI', 'ARI', 'ACC', 'Silhouette'],
        index=['NMI', 'ARI', 'ACC', 'Silhouette'].index(val_opt.get('lr_optimization_metric', 'NMI'))
        if val_opt.get('lr_optimization_metric', 'NMI') in ['NMI', 'ARI', 'ACC', 'Silhouette'] else 0,
        key=f"lr_opt_metric_{selected_idx}",
        help="Metric used to pick the best LR on validation."
      )
    with col_lr2:
      lr_repeats = st.number_input(
        "Repeats per LR candidate",
        min_value=1,
        max_value=5,
        value=int(val_opt.get('lr_optimization_repeats', 1)),
        key=f"lr_opt_repeats_{selected_idx}",
        help="Repeat each candidate this many times during LR optimization."
      )

    default_scales = val_opt.get('lr_optimization_scales', [100.0, 10.0, 1.0, 0.1])
    scales_str = st.text_input(
      "LR Scales (comma-separated)",
      value=",".join(str(x) for x in default_scales),
      key=f"lr_opt_scales_{selected_idx}",
      help="Multipliers around current LR, e.g. 100,10,1,0.1."
    )
    parsed_scales = _parse_float_list(scales_str)
    if not parsed_scales:
      st.warning("Invalid LR scales. Falling back to 100,10,1,0.1.")
      parsed_scales = [100.0, 10.0, 1.0, 0.1]

    current_config['validation_optimization'] = {
      'enable_lr_optimization': bool(enable_lr_opt),
      'lr_optimization_metric': lr_metric,
      'lr_optimization_repeats': int(lr_repeats),
      'lr_optimization_scales': parsed_scales
    }

  if render_protocol_workbench is not None:
    render_protocol_workbench(
      st.session_state.custom_benchmarks,
      selected_idx,
      _build_commands_for_config,
    )

  # -------------------------------------------------------------------------
  # Generate Output
  # -------------------------------------------------------------------------
  st.markdown("---")
  st.subheader("Generate Scripts")
  
  if st.button("Generate CLI Commands for All Configurations", type="primary"):
    st.success(f"Generated commands for {len(st.session_state.custom_benchmarks)} configurations.")
    
    all_commands = []
    for i, config in enumerate(st.session_state.custom_benchmarks):
      # Generate command using the refactored function, passing the config dict
      try:
        commands_for_config = _build_commands_for_config(config)

        if not commands_for_config:
          st.warning(f"No local algorithms, report methods, or manual protocols selected for '{config['name']}'.")
          continue
        
        expander_title = f"{i+1}. {config['name']}"
        with st.expander(expander_title, expanded=True):
          st.code("\n\n".join(commands_for_config), language="bash")
        
        # Add to bulk script
        all_commands.append(f"# Configuration: {config['name']}\n" + "\n\n".join(commands_for_config) + "\n")
        
      except Exception as e:
        st.error(f"Error generating command for '{config['name']}': {e}")

    # Download All
    full_script = "#!/bin/bash\n# Batch Benchmark Script\n# Generated by SCRBenchmark UI\n\n" + "\n".join(all_commands)
    st.download_button(
      "Download Full Batch Script",
      full_script,
      file_name="run_batch_benchmark.sh",
      mime="text/x-shellscript"
    )

def _render_custom_algo_params(algo_name: str, config: Dict[str, Any], config_idx: int):
  """Render hyperparameter editors for a specific algorithm within a custom config."""
  from core.config import ParamType

  algo_class = AlgorithmRegistry.get(algo_name)
  hyperparams = algo_class.get_hyperparameters()
  algo_info = algo_class.get_info()

  if algo_name not in config['algorithm_params']:
    config['algorithm_params'][algo_name] = {hp.name: hp.default for hp in hyperparams}
    # Use algorithm's recommended_data field for default input type.
    # Classical PCA-based algorithms recommend preprocessed data; deep models
    # typically recommend raw counts for count-aware losses.
    recommended = getattr(algo_info, 'recommended_data', 'raw')
    # Map to UI-compatible values: 'processed' or 'raw_filtered'
    if recommended in ['preprocessed', 'processed']:
      input_type = 'processed'
    else:
      # 'raw', 'raw_filtered', or any other value -> use raw_filtered
      input_type = 'raw_filtered'
    config['algorithm_params'][algo_name]['input_type'] = input_type
    config['algorithm_params'][algo_name]['use_raw_data'] = (input_type == 'raw_filtered')
    
  params = config['algorithm_params'][algo_name]
  
  # 1. Input Data Type
  input_options = ['processed', 'raw_filtered']
  input_labels = {
    'processed': 'Processed (LogNorm + Scaled + HVG)',
    'raw_filtered': 'Raw Counts (QC-Filtered + HVG)',
  }
  
  current_input = params.get('input_type', 'raw_filtered')
  if current_input not in input_options: current_input = 'raw_filtered'
  
  selected_input = st.selectbox(
    "Input Data Type",
    options=input_options,
    index=input_options.index(current_input),
    format_func=lambda x: input_labels[x],
    key=f"input_type_{config_idx}_{algo_name}"
  )
  params['input_type'] = selected_input
  params['use_raw_data'] = (selected_input == 'raw_filtered')

  # 2. Hyperparameters
  for hp in hyperparams:
    key = f"hp_{config_idx}_{algo_name}_{hp.name}"
    current_val = params.get(hp.name, hp.default)
    
    if hp.param_type == ParamType.INTEGER:
      params[hp.name] = st.number_input(
        hp.display_name,
        min_value=hp.min_value if hp.min_value is not None else 0,
        max_value=hp.max_value if hp.max_value is not None else 100000,
        value=int(current_val),
        help=hp.description,
        key=key
      )
    elif hp.param_type == ParamType.FLOAT:
      params[hp.name] = st.number_input(
        hp.display_name,
        min_value=float(hp.min_value) if hp.min_value is not None else 0.0,
        max_value=float(hp.max_value) if hp.max_value is not None else 1.0,
        value=float(current_val),
        format="%.4f",
        help=hp.description,
        key=key
      )
    elif hp.param_type == ParamType.BOOLEAN:
      params[hp.name] = st.checkbox(
        hp.display_name,
        value=bool(current_val),
        help=hp.description,
        key=key
      )
    elif hp.param_type == ParamType.CHOICE:
      choices = hp.choices or []
      if choices:
        params[hp.name] = st.selectbox(
          hp.display_name,
          options=choices,
          index=choices.index(current_val) if current_val in choices else 0,
          help=hp.description,
          key=key
        )
    elif hp.param_type == ParamType.STRING:
      if hp.choices:
        params[hp.name] = st.selectbox(
          hp.display_name,
          options=hp.choices,
          index=hp.choices.index(current_val) if current_val in hp.choices else 0,
          help=hp.description,
          key=key
        )
      else:
        params[hp.name] = st.text_input(
          hp.display_name,
          value=str(current_val),
          help=hp.description,
          key=key
        )


def _build_algo_modifications_summary(config: Dict[str, Any], selected_algorithms: List[str]) -> pd.DataFrame:
  """Build a compact summary of per-algorithm modified hyperparameters."""
  rows = []
  algo_params = config.get('algorithm_params', {}) or {}

  for algo_name in selected_algorithms:
    algo_class = AlgorithmRegistry.get(algo_name)
    if algo_class is None:
      continue

    info = algo_class.get_info()
    hyperparams = algo_class.get_hyperparameters()
    defaults = {hp.name: hp.default for hp in hyperparams}
    current = algo_params.get(algo_name, {}) or {}

    modified = []
    for param_name, value in current.items():
      # input_type is a UI-level option, not part of HyperparameterConfig defaults.
      if param_name == 'input_type':
        continue
      if param_name not in defaults:
        continue
      if value != defaults[param_name]:
        modified.append(param_name)

    rows.append({
      'Algorithm': info.display_name,
      'Modified Params': len(modified),
      'Parameters': ', '.join(modified[:8]) + ('...' if len(modified) > 8 else '')
    })

  return pd.DataFrame(rows)


def _parse_float_list(raw: str) -> List[float]:
  """Parse comma-separated positive float list."""
  if raw is None:
    return []
  values = []
  for tok in str(raw).split(','):
    tok = tok.strip()
    if not tok:
      continue
    try:
      val = float(tok)
      if val > 0:
        values.append(val)
    except Exception:
      continue
  return values


def _parse_balance_target(raw: Any) -> Any:
  """
  Parse balancing target from UI input.

  Returns:
    - 'auto' for auto-detection
    - positive int
    - None if invalid
  """
  if raw is None:
    return 'auto'

  if isinstance(raw, int):
    return raw if raw > 0 else None

  text = str(raw).strip().lower()
  if text in ['', 'auto', 'none']:
    return 'auto'

  try:
    value = int(text)
    return value if value > 0 else None
  except Exception:
    return None

def _create_default_config(name: str) -> Dict[str, Any]:
  """Create a default benchmark configuration dict."""
  return {
    'name': name,
    'uploaded_file_path': 'data/your_data.h5ad',
    'dataset_key': 'your_data',
    'label_key': 'Group',
    'batch_key': 'batch',
    'n_labels': 0,
    'selected_algorithms': ['scdeepcluster'],
    'selected_report_methods': [],
    'scraw_preset': 'default',
    'report_include_harmony_variants': False,
    'report_harmony_variant_methods': [],
    'report_method_params': '',
    'report_method_n_pcs': 50,
    'report_method_harmony_max_iter': 10,
    'report_method_harmony_nclust': 50,
    'preprocessing_params': {
      # Quality filtering
      'do_cell_filtering': True,
      'do_gene_filtering': True,
      'min_genes_per_cell': 100,
      'max_genes_per_cell': 10000,
      'min_cells_per_gene': 3,
      # Normalization
      'do_normalization': True,
      'do_log_transform': True,
      'target_sum': 20000,
      # Feature selection
      'do_hvg': True,
      'n_top_genes': 2000,
      'hvg_flavor': 'seurat',
      'hvg_strategy': 'train_only',
      # Scaling
      'do_scaling': True,
      'scale_max_value': 10.0,
      # Batch correction
      'do_batch_correction': False,
      'batch_correction_method': 'scvi',
      'batch_correction_batch_key': 'batch'
    },
    'benchmark_configured': True,
    'benchmark_setup': {
      'mode': 'benchmark',
      'original_settings': {
        'mode': 'stratified',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'stratify_by_batch': True,
        'stratify_by_labels': True,
        'use_stratified_base': True,
        'batch_col': 'batch'
      }
    },
    'balance_settings_standard': {}, # Batch balancing for standard mode (no split)
    'global_network_config': {
      'enabled': False,
      'encoder_layers': '256,64',
      'z_dim': 32
    },
    'algorithm_params': {}, # Default empty, will use algorithm defaults
    'device_preference': 'auto',
    'n_repeats': 1,
    'seed': 42,
    'validation_optimization': {
      'enable_lr_optimization': True,
      'lr_optimization_metric': 'NMI',
      'lr_optimization_repeats': 1,
      'lr_optimization_scales': [100.0, 10.0, 1.0, 0.1]
    },
    'manual_protocols': {
      'enabled': False,
      'selected_protocols': [],
      'seeds': '42',
      'scib_n_jobs': 4,
      'overwrite': False,
      'verbose': True,
      'extra_params': '',
      'loss_transfer': {
        'methods': ['scMAE', 'scDeepCluster', 'DESC'],
        'variants': ['baseline', 'weighted'],
        'weight_params': ''
      },
      'inductive': {
        'algorithms': ['scraw', 'scname', 'sc_mae', 'scdeepcluster'],
        'split_key': 'batch',
        'train_split_key': '',
        'test_split_key': '',
        'train_batches': '',
        'test_batches': '',
        'preset': 'default',
        'trial_config_path': '',
        'baseline_runtime_profile': 'scrbenchmark-default',
        'skip_existing': False
      }
    },
    'output_dir': 'results',
    'save_labels': True,
    'save_embeddings': False
  }
