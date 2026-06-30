"""Report reproduction launchers and traceability page."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = REPO_ROOT / "reproducibility" / "report_reproduction_map.csv"
DOC_PATH = REPO_ROOT / "docs" / "report_reproduction_map.md"


def _quote(parts: Iterable[object]) -> str:
  return " ".join(shlex.quote(str(part)) for part in parts)


def _run_generator(cmd: list[object]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    [str(part) for part in cmd],
    cwd=REPO_ROOT,
    text=True,
    capture_output=True,
    check=False,
  )


def _read_csv(path: Path) -> pd.DataFrame | None:
  if not path.exists():
    return None
  try:
    return pd.read_csv(path)
  except Exception:
    return None


def _resolve_repo_path(raw_path: object) -> Path:
  path = Path(str(raw_path)).expanduser()
  return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _show_generated_artifacts(plan_path: Path, shell_path: Path) -> None:
  cols = st.columns([1, 1])
  with cols[0]:
    if plan_path.exists():
      st.success(f"Plan written: {plan_path}")
    else:
      st.warning(f"Plan not found: {plan_path}")
  with cols[1]:
    if shell_path.exists():
      st.success(f"Launcher written: {shell_path}")
    else:
      st.warning(f"Launcher not found: {shell_path}")

  frame = _read_csv(plan_path)
  if frame is not None:
    st.dataframe(frame.head(50), use_container_width=True)
    st.caption(f"{len(frame)} planned job(s)")

  if shell_path.exists():
    st.code(f"bash {shell_path}", language="bash")
    st.download_button(
      "Download launcher",
      data=shell_path.read_text(encoding="utf-8"),
      file_name=shell_path.name,
      mime="text/x-shellscript",
      width="stretch",
    )


def _render_traceability_tab() -> None:
  st.subheader("Traceability Matrix")
  st.info(
    "This tab shows only report figures and tables that are directly "
    "rerunnable from the panel. It does not run experiments. Use Stable "
    "Generalist, Report Complements, or Custom Protocols to generate runnable "
    "shell scripts."
  )
  frame = _read_csv(MAP_PATH)
  if frame is None:
    st.error(f"Could not read {MAP_PATH}")
    return

  filtered = frame[frame["coverage"] == "rerunnable"].copy()

  st.caption(
    "Each row is one report figure or table label with a complete launcher. "
    "Partial, archival, documented-only, and not-yet-integrated entries are "
    "hidden from this view."
  )
  shown_cols = [
    "report_label",
    "report_object",
    "plan_generator",
    "campaigns",
    "datasets",
    "seeds",
    "launcher",
    "notes",
  ]
  st.dataframe(filtered[shown_cols], use_container_width=True, hide_index=True)
  st.caption(f"Documentation: {DOC_PATH}")


def _render_stable_generalist_tab() -> None:
  st.subheader("Stable-Generalist Plan")
  st.caption(
    "Use this tab to regenerate the main benchmark plan for the common-8 and "
    "external validation report results. It writes planned_jobs.csv and "
    "run_ready_jobs.sh."
  )
  st.info(
    "Dataset and method filters are inclusion filters: leave them empty to "
    "evaluate everything from the reference table, or fill them to keep only "
    "the selected datasets/methods."
  )
  col1, col2 = st.columns(2)
  with col1:
    output_root = st.text_input(
      "Output root",
      value="results/stable_generalist_repro",
      key="report_repro_stable_output",
    )
    python_bin = st.text_input(
      "Python interpreter",
      value=sys.executable,
      key="report_repro_stable_python",
    )
    device = st.selectbox("Device", ["cuda", "cpu", "auto", "mps"], key="report_repro_stable_device")
  with col2:
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, key="report_repro_stable_seed")
    datasets = st.text_input(
      "Datasets to evaluate",
      value="",
      placeholder="baron_human_pancreas,kang_pbmc_gse96583_singlets_raw_counts",
      key="report_repro_stable_datasets",
      help=(
        "Optional comma-separated inclusion filter. Empty means all datasets; "
        "a value means generate jobs only for these dataset keys."
      ),
    )
    methods = st.text_input(
      "Methods to evaluate",
      value="",
      placeholder="scRAW,scMAE,Harmony",
      key="report_repro_stable_methods",
      help=(
        "Optional comma-separated inclusion filter. Empty means all methods; "
        "a value means generate jobs only for these method names."
      ),
    )
  strict_data = st.checkbox("Mark missing data as blocked", value=False, key="report_repro_stable_strict")

  cmd: list[object] = [
    python_bin,
    "scripts/reproduction/build_stable_generalist_plan.py",
    "--output-root",
    output_root,
    "--python-bin",
    python_bin,
    "--device",
    device,
    "--seed",
    int(seed),
  ]
  if datasets.strip():
    cmd.extend(["--datasets", datasets.strip()])
  if methods.strip():
    cmd.extend(["--methods", methods.strip()])
  if strict_data:
    cmd.append("--strict-data")

  st.code(_quote(cmd), language="bash")
  if st.button("Generate stable-generalist launcher", type="primary", width="stretch"):
    result = _run_generator(cmd)
    if result.returncode == 0:
      st.success("Plan generated")
    else:
      st.error("Plan generation failed")
    if result.stdout:
      st.code(result.stdout, language="text")
    if result.stderr:
      st.code(result.stderr, language="text")

  root = _resolve_repo_path(output_root)
  _show_generated_artifacts(root / "planned_jobs.csv", root / "run_ready_jobs.sh")


def _render_report_plan_tab() -> None:
  st.subheader("Report Complement Plan")
  st.caption(
    "Use this tab for report complements: inductive splits, loss-transfer "
    "experiments, and DEG marker-overlap analysis. It writes "
    "report_planned_jobs.csv and run_ready_report_jobs.sh."
  )
  col1, col2 = st.columns(2)
  with col1:
    output_root = st.text_input("Output root", value="results/report_repro", key="report_repro_report_output")
    python_bin = st.text_input("Python interpreter", value=sys.executable, key="report_repro_report_python")
    device = st.selectbox("Device", ["cuda", "cpu", "auto", "mps"], key="report_repro_report_device")
  with col2:
    campaigns = st.multiselect(
      "Campaigns",
      ["inductive", "loss_transfer", "deg"],
      default=["inductive", "loss_transfer", "deg"],
      key="report_repro_report_campaigns",
    )
    strict_data = st.checkbox("Mark missing data as blocked", value=False, key="report_repro_report_strict")
    reuse_existing = st.checkbox(
      "Reuse existing scRAW Baron labels when available",
      value=True,
      key="report_repro_report_reuse_existing",
      help=(
        "The DEG/marker-overlap campaign can reuse the existing Baron scRAW "
        "labels from the report artifact folder instead of planning a new "
        "source scRAW run."
      ),
    )

  cmd: list[object] = [
    python_bin,
    "scripts/reproduction/build_report_plan.py",
    "--output-root",
    output_root,
    "--python-bin",
    python_bin,
    "--device",
    device,
    "--campaigns",
    ",".join(campaigns) if campaigns else "all",
  ]
  if strict_data:
    cmd.append("--strict-data")
  if not reuse_existing:
    cmd.append("--no-reuse-existing-artifacts")

  st.code(_quote(cmd), language="bash")
  if st.button("Generate report launcher", type="primary", width="stretch"):
    result = _run_generator(cmd)
    if result.returncode == 0:
      st.success("Plan generated")
    else:
      st.error("Plan generation failed")
    if result.stdout:
      st.code(result.stdout, language="text")
    if result.stderr:
      st.code(result.stderr, language="text")

  root = _resolve_repo_path(output_root)
  _show_generated_artifacts(root / "report_planned_jobs.csv", root / "run_ready_report_jobs.sh")


def _common_manual_inputs(
  prefix: str = "manual_repro",
  *,
  output_default: str = "results/manual_report_protocol",
  script_default: str = "results/manual_report_protocol/run_jobs.sh",
) -> dict[str, object]:
  col1, col2, col3 = st.columns(3)
  with col1:
    data = st.text_input(
      "Data",
      value="data/stable_generalist/baron_human_pancreas.h5ad",
      key=f"{prefix}_data",
    )
    dataset_key = st.text_input("Dataset key", value="baron_human_pancreas", key=f"{prefix}_dataset")
    label_key = st.text_input("Label key", value="label", key=f"{prefix}_label")
  with col2:
    batch_key = st.text_input("Batch key", value="batch", key=f"{prefix}_batch")
    n_labels = st.number_input("Number of labels", min_value=0, max_value=1000, value=14, key=f"{prefix}_n_labels")
    seeds = st.text_input("Seeds", value="42", key=f"{prefix}_seeds")
  with col3:
    output_root = st.text_input("Output root", value=output_default, key=f"{prefix}_output")
    script = st.text_input("Launcher path", value=script_default, key=f"{prefix}_script")
    python_bin = st.text_input("Python interpreter", value=sys.executable, key=f"{prefix}_python")
  device = st.selectbox("Device", ["cuda", "cpu", "auto", "mps"], key=f"{prefix}_device")
  return {
    "data": data,
    "dataset_key": dataset_key,
    "label_key": label_key,
    "batch_key": batch_key,
    "n_labels": int(n_labels),
    "seeds": seeds,
    "output_root": output_root,
    "script": script,
    "python_bin": python_bin,
    "device": device,
  }


def _base_manual_protocol_cmd(protocol: str, common: dict[str, object]) -> list[object]:
  return [
    common["python_bin"],
    "scripts/reproduction/manual_protocols.py",
    "--protocol",
    protocol,
    "--data",
    common["data"],
    "--output-root",
    common["output_root"],
    "--dataset-key",
    common["dataset_key"],
    "--label-key",
    common["label_key"],
    "--batch-key",
    common["batch_key"],
    "--n-labels",
    common["n_labels"],
    "--seeds",
    common["seeds"],
    "--device",
    common["device"],
    "--python-bin",
    common["python_bin"],
    "--script",
    common["script"],
  ]


def _show_manual_launcher(cmd: list[object], common: dict[str, object], *, button_label: str, key: str) -> None:
  st.code(_quote(cmd), language="bash")
  if st.button(button_label, type="primary", width="stretch", key=key):
    result = _run_generator(cmd)
    if result.returncode == 0:
      st.success("Launcher generated")
    else:
      st.error("Launcher generation failed")
    if result.stdout:
      st.code(result.stdout, language="text")
    if result.stderr:
      st.code(result.stderr, language="text")

  root = _resolve_repo_path(common["output_root"])
  shell_path = _resolve_repo_path(common["script"])
  _show_generated_artifacts(root / "manual_protocol_jobs.csv", shell_path)


def _render_loss_transfer_tab() -> None:
  st.subheader("scRAW Weighting Loss Transfer")
  st.caption(
    "Generate commands that plug scRAW cell weighting into existing algorithms. "
    "Use the selectors to choose which implemented methods and weighting variants "
    "are included."
  )
  common = _common_manual_inputs(
    "loss_transfer_repro",
    output_default="results/manual_loss_transfer",
    script_default="results/manual_loss_transfer/run_jobs.sh",
  )
  col1, col2 = st.columns(2)
  with col1:
    methods = st.multiselect(
      "Algorithms",
      ["scMAE", "scDeepCluster", "DESC"],
      default=["scMAE", "scDeepCluster", "DESC"],
      key="loss_transfer_repro_methods",
    )
  with col2:
    variants = st.multiselect(
      "Weighting variants",
      ["baseline", "weighted", "density_only", "kmeans", "triplet"],
      default=["baseline", "weighted"],
      key="loss_transfer_repro_variants",
    )
  cmd = _base_manual_protocol_cmd("loss_transfer", common)
  cmd.extend(["--loss-methods", ",".join(methods), "--loss-variants", ",".join(variants)])
  _show_manual_launcher(
    cmd,
    common,
    button_label="Generate loss-transfer launcher",
    key="loss_transfer_repro_generate",
  )


def _render_generalization_tab() -> None:
  st.subheader("Generalization / Inductive Protocol")
  st.caption(
    "Generate train/test-group commands for existing inductive algorithms. "
    "This is the report complement used to assess generalization across batches."
  )
  common = _common_manual_inputs(
    "generalization_repro",
    output_default="results/manual_inductive",
    script_default="results/manual_inductive/run_jobs.sh",
  )
  algorithms = st.multiselect(
    "Algorithms",
    ["scraw", "scname", "sc_mae", "scdeepcluster", "scaide", "pca_harmony"],
    default=["scraw", "scname", "sc_mae", "scdeepcluster"],
    key="generalization_repro_algorithms",
  )
  col1, col2, col3 = st.columns(3)
  with col1:
    split_key = st.text_input("Split key", value="batch", key="generalization_repro_split_key")
  with col2:
    train_batches = st.text_input(
      "Train groups",
      value="human1,human2,human3",
      key="generalization_repro_train",
    )
  with col3:
    test_batches = st.text_input("Test groups", value="human4", key="generalization_repro_test")
  cmd = _base_manual_protocol_cmd("inductive", common)
  cmd.extend(
    [
      "--inductive-algorithms",
      ",".join(algorithms),
      "--split-key",
      split_key,
      "--train-batches",
      train_batches,
      "--test-batches",
      test_batches,
    ]
  )
  _show_manual_launcher(
    cmd,
    common,
    button_label="Generate generalization launcher",
    key="generalization_repro_generate",
  )


def _render_biological_interpretation_tab() -> None:
  st.subheader("Biological Interpretation / Marker Overlap")
  st.caption(
    "Generate the Baron scRAW marker-overlap analysis. Existing scRAW labels "
    "are reused by default when the local report artifacts are present; this "
    "avoids rerunning the source model."
  )
  col1, col2 = st.columns(2)
  with col1:
    output_root = st.text_input("Output root", value="results/report_repro", key="bio_repro_output")
    python_bin = st.text_input("Python interpreter", value=sys.executable, key="bio_repro_python")
  with col2:
    device = st.selectbox("Device", ["cuda", "cpu", "auto", "mps"], key="bio_repro_device")
    strict_data = st.checkbox("Mark missing data as blocked", value=False, key="bio_repro_strict")
    reuse_existing = st.checkbox("Reuse existing scRAW Baron labels", value=True, key="bio_repro_reuse_existing")

  cmd: list[object] = [
    python_bin,
    "scripts/reproduction/build_report_plan.py",
    "--output-root",
    output_root,
    "--python-bin",
    python_bin,
    "--device",
    device,
    "--campaigns",
    "deg",
  ]
  if strict_data:
    cmd.append("--strict-data")
  if not reuse_existing:
    cmd.append("--no-reuse-existing-artifacts")

  st.code(_quote(cmd), language="bash")
  if st.button("Generate marker-overlap launcher", type="primary", width="stretch", key="bio_repro_generate"):
    result = _run_generator(cmd)
    if result.returncode == 0:
      st.success("Plan generated")
    else:
      st.error("Plan generation failed")
    if result.stdout:
      st.code(result.stdout, language="text")
    if result.stderr:
      st.code(result.stderr, language="text")

  root = _resolve_repo_path(output_root)
  _show_generated_artifacts(root / "report_planned_jobs.csv", root / "run_ready_report_jobs.sh")


def _render_manual_protocol_tab() -> None:
  st.subheader("Custom Protocol Launcher")
  st.caption(
    "Use this tab to design new report-style commands by changing the dataset, "
    "methods, seeds, or train/test groups. It writes a custom planned-job CSV "
    "and shell launcher."
  )
  protocol = st.selectbox("Protocol", ["loss_transfer", "harmony", "inductive"], key="manual_repro_protocol")
  common = _common_manual_inputs()

  cmd: list[object] = _base_manual_protocol_cmd(protocol, common)

  if protocol == "loss_transfer":
    col1, col2 = st.columns(2)
    with col1:
      methods = st.multiselect(
        "Loss methods",
        ["scMAE", "scDeepCluster", "DESC"],
        default=["scMAE", "scDeepCluster", "DESC"],
        key="manual_repro_loss_methods",
      )
    with col2:
      variants = st.multiselect(
        "Loss variants",
        ["baseline", "weighted", "density_only", "kmeans", "triplet"],
        default=["baseline", "weighted", "triplet"],
        key="manual_repro_loss_variants",
      )
    cmd.extend(["--loss-methods", ",".join(methods), "--loss-variants", ",".join(variants)])
  elif protocol == "harmony":
    methods = st.text_input(
      "Harmony methods",
      value="Harmony,scMAE+Harmony,scNAME+Harmony,scvi+Harmony",
      key="manual_repro_harmony_methods",
    )
    cmd.extend(["--harmony-methods", methods])
  else:
    algorithms = st.text_input(
      "Inductive algorithms",
      value="scraw,scname,sc_mae,scdeepcluster,pca_harmony",
      key="manual_repro_inductive_algorithms",
    )
    split_key = st.text_input("Split key", value="batch", key="manual_repro_split_key")
    train_batches = st.text_input("Train groups", value="human1,human2,human3", key="manual_repro_train")
    test_batches = st.text_input("Test groups", value="human4", key="manual_repro_test")
    cmd.extend(
      [
        "--inductive-algorithms",
        algorithms,
        "--split-key",
        split_key,
        "--train-batches",
        train_batches,
        "--test-batches",
        test_batches,
      ]
    )

  _show_manual_launcher(cmd, common, button_label="Generate custom launcher", key="manual_repro_generate")


def render_report_reproduction_page() -> None:
  """Render the report reproduction page."""
  st.header("Report Reproduction")
  st.markdown(
    "This page helps reproduce the internship report experiments. Start with "
    "**Traceability** to see which report figure/table is covered, then use one "
    "of the launcher tabs to generate the shell script that runs the jobs."
  )
  st.warning(
    "Generating a launcher is lightweight. Running the generated shell script "
    "can start long GPU/CPU experiments."
  )

  tabs = st.tabs([
    "Traceability",
    "Stable Generalist",
    "Report Complements",
    "scRAW Weighting",
    "Generalization",
    "Biological Interpretation",
    "Custom Protocols",
  ])
  with tabs[0]:
    _render_traceability_tab()
  with tabs[1]:
    _render_stable_generalist_tab()
  with tabs[2]:
    _render_report_plan_tab()
  with tabs[3]:
    _render_loss_transfer_tab()
  with tabs[4]:
    _render_generalization_tab()
  with tabs[5]:
    _render_biological_interpretation_tab()
  with tabs[6]:
    _render_manual_protocol_tab()
