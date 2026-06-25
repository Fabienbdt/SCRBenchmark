"""Streamlit widgets for protocol-driven benchmark design."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import streamlit as st

try:
  from protocols.registry import (
    build_job_plan,
    collect_result_rows,
    expand_sweep_configs,
    load_protocol_specs,
    protocol_to_customize_configs,
    protocol_to_yaml,
    run_plan_job,
    summarize_results,
    validate_customize_config,
    write_protocol_artifacts,
  )
except Exception:
  build_job_plan = None
  collect_result_rows = None
  expand_sweep_configs = None
  load_protocol_specs = None
  protocol_to_customize_configs = None
  protocol_to_yaml = None
  run_plan_job = None
  summarize_results = None
  validate_customize_config = None
  write_protocol_artifacts = None


def _registry_available() -> bool:
  return all(
    item is not None
    for item in [
      build_job_plan,
      collect_result_rows,
      expand_sweep_configs,
      load_protocol_specs,
      protocol_to_customize_configs,
      validate_customize_config,
      write_protocol_artifacts,
    ]
  )


def _rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
  return pd.DataFrame([dict(row) for row in rows])


def _protocol_metadata(spec: Any) -> dict[str, Any]:
  raw = getattr(spec, "raw", {}) or {}
  return {
    "id": getattr(spec, "id", ""),
    "name": getattr(spec, "name", ""),
    "type": getattr(spec, "experiment_type", ""),
    "tags": ", ".join(getattr(spec, "tags", [])),
    "source": str(getattr(spec, "path", "") or ""),
    "datasets": len(raw.get("datasets", []) or ([raw.get("dataset")] if raw.get("dataset") else [])),
    "inductive_splits": len(raw.get("inductive_splits", []) or []),
  }


def _sweep_text_from_config(config: Mapping[str, Any]) -> str:
  sweeps = config.get("sweep_params", {}) or {}
  if not sweeps:
    return ""
  lines = []
  for key, values in sweeps.items():
    if isinstance(values, (list, tuple)):
      value_text = ",".join(str(value) for value in values)
    else:
      value_text = str(values)
    lines.append(f"{key}={value_text}")
  return "\n".join(lines)


def render_protocol_registry_panel(selected_idx: int) -> None:
  """Render YAML protocol loader controls."""

  if not _registry_available():
    st.info("Protocol registry unavailable in this environment.")
    return

  with st.expander("Versioned Protocol Registry", expanded=False):
    specs = load_protocol_specs()
    if not specs:
      st.warning("No protocol YAML files found under protocols/.")
      return

    protocol_ids = sorted(specs)
    selected_protocol_id = st.selectbox(
      "Protocol preset",
      protocol_ids,
      format_func=lambda protocol_id: f"{specs[protocol_id].name} [{protocol_id}]",
      key="versioned_protocol_selector",
    )
    spec = specs[selected_protocol_id]

    meta = _protocol_metadata(spec)
    st.dataframe(pd.DataFrame([meta]), hide_index=True, use_container_width=True)
    if spec.description:
      st.caption(spec.description)

    limitations = list((spec.raw or {}).get("limitations", []) or [])
    for limitation in limitations:
      st.warning(str(limitation))

    if protocol_to_yaml is not None:
      st.download_button(
        "Download YAML",
        data=protocol_to_yaml(spec),
        file_name=f"{selected_protocol_id}.yaml",
        mime="text/yaml",
      )

    col_add, col_replace = st.columns(2)
    with col_add:
      if st.button("Add Protocol Configs", use_container_width=True):
        new_configs = protocol_to_customize_configs(spec)
        if not new_configs:
          st.warning("No configuration generated for this protocol.")
        else:
          st.session_state.custom_benchmarks.extend(new_configs)
          st.session_state.current_config_idx = len(st.session_state.custom_benchmarks) - len(new_configs)
          st.rerun()
    with col_replace:
      if st.button("Replace Current With Protocol", use_container_width=True):
        new_configs = protocol_to_customize_configs(spec)
        if not new_configs:
          st.warning("No configuration generated for this protocol.")
        else:
          st.session_state.custom_benchmarks[selected_idx:selected_idx + 1] = new_configs
          st.session_state.current_config_idx = selected_idx
          st.rerun()


def _render_validation(configs: Sequence[Mapping[str, Any]]) -> None:
  st.markdown("#### Validation")
  check_data = st.checkbox(
    "Check dataset files and AnnData columns",
    value=True,
    key="protocol_validation_check_data",
  )

  rows = []
  ok_count = 0
  for idx, config in enumerate(configs, start=1):
    result = validate_customize_config(config, check_data=check_data)
    if result.ok:
      ok_count += 1
    for row in result.rows():
      rows.append({
        "config": idx,
        "name": config.get("name", ""),
        **row,
      })

  c1, c2, c3 = st.columns(3)
  c1.metric("Configs", len(configs))
  c2.metric("Valid", ok_count)
  c3.metric("Issues", len([row for row in rows if row["severity"] in {"error", "warning"}]))

  if rows:
    frame = _rows_frame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
  else:
    st.success("All configurations validate without issues.")


def _render_sweeps(configs: list[dict[str, Any]], selected_idx: int) -> None:
  st.markdown("#### Sweeps")
  current = configs[selected_idx]
  default_text = _sweep_text_from_config(current)
  sweep_text = st.text_area(
    "Sweep definition for current config",
    value=default_text,
    key=f"protocol_sweep_text_{selected_idx}",
    height=110,
    placeholder="preprocessing.n_top_genes=1000,2000\nalgorithm.sc_mae.lr=0.001,0.0005",
    help=(
      "One path=value list per line. Supported aliases include execution.seed, "
      "execution.n_repeats, preprocessing.*, report_method.*, manual.*, and algorithm.<name>.<param>."
    ),
  )

  col_preview, col_append, col_replace = st.columns(3)
  with col_preview:
    if st.button("Preview Sweep", use_container_width=True):
      try:
        expanded = expand_sweep_configs(current, sweep_text)
        st.session_state.protocol_sweep_preview = expanded
      except Exception as exc:
        st.error(f"Sweep expansion failed: {exc}")
  with col_append:
    if st.button("Append Sweep Variants", use_container_width=True):
      try:
        expanded = expand_sweep_configs(current, sweep_text)
        if len(expanded) <= 1:
          st.info("Sweep has one configuration; nothing appended.")
        else:
          configs.extend(expanded)
          st.session_state.current_config_idx = len(configs) - len(expanded)
          st.rerun()
      except Exception as exc:
        st.error(f"Sweep expansion failed: {exc}")
  with col_replace:
    if st.button("Replace With Sweep Variants", use_container_width=True):
      try:
        expanded = expand_sweep_configs(current, sweep_text)
        configs[selected_idx:selected_idx + 1] = expanded
        st.session_state.current_config_idx = selected_idx
        st.rerun()
      except Exception as exc:
        st.error(f"Sweep expansion failed: {exc}")

  preview = st.session_state.get("protocol_sweep_preview")
  if preview:
    st.caption(f"Sweep preview: {len(preview)} configuration(s).")
    st.dataframe(
      pd.DataFrame([
        {
          "name": config.get("name", ""),
          "output_dir": config.get("output_dir", ""),
          "assignment": config.get("sweep_assignment", {}),
        }
        for config in preview
      ]),
      hide_index=True,
      use_container_width=True,
    )


def _render_job_plan(
  configs: Sequence[Mapping[str, Any]],
  command_builder: Callable[[Mapping[str, Any]], Sequence[str]],
) -> None:
  st.markdown("#### Job Plan")
  plan_root = st.text_input(
    "Plan output directory",
    value=st.session_state.get("protocol_plan_root", "results/protocol_design/latest_plan"),
    key="protocol_plan_root",
  )
  plan_name = st.text_input(
    "Plan name",
    value=st.session_state.get("protocol_plan_name", "custom_protocol"),
    key="protocol_plan_name",
  )

  col_build, col_write = st.columns(2)
  with col_build:
    if st.button("Build Job Plan", type="primary", use_container_width=True):
      try:
        st.session_state.protocol_job_plan = build_job_plan(configs, command_builder)
        st.session_state.protocol_job_status = {}
        st.session_state.protocol_job_logs = {}
      except Exception as exc:
        st.error(f"Could not build job plan: {exc}")
  rows = st.session_state.get("protocol_job_plan", [])
  with col_write:
    if st.button("Write Plan Artifacts", use_container_width=True, disabled=not bool(rows)):
      try:
        paths = write_protocol_artifacts(rows, plan_root, configs, name=plan_name)
        st.session_state.protocol_artifact_paths = paths
        st.success("Plan artifacts written.")
      except Exception as exc:
        st.error(f"Could not write plan artifacts: {exc}")

  if st.session_state.get("protocol_artifact_paths"):
    paths = st.session_state.protocol_artifact_paths
    st.json(paths)

  if not rows:
    st.info("Build the job plan to inspect commands, write artifacts, or execute selected jobs.")
    return

  frame = _rows_frame(rows)
  st.dataframe(frame.drop(columns=["command"], errors="ignore"), hide_index=True, use_container_width=True)
  with st.expander("Commands", expanded=False):
    st.code("\n\n".join(row["command"] for row in rows), language="bash")


def _render_execution() -> None:
  rows = st.session_state.get("protocol_job_plan", [])
  if not rows or run_plan_job is None:
    return

  st.markdown("#### Execution")
  status_map = st.session_state.setdefault("protocol_job_status", {})
  for row in rows:
    status_map.setdefault(row["job_id"], row.get("status", "pending"))

  selectable = [row["job_id"] for row in rows]
  default_jobs = selectable[:1]
  selected_jobs = st.multiselect(
    "Jobs to run",
    selectable,
    default=default_jobs,
    key="protocol_jobs_to_run",
  )
  log_dir = st.text_input(
    "Execution log directory",
    value=st.session_state.get("protocol_log_dir", "results/protocol_design/logs"),
    key="protocol_log_dir",
  )

  col_run, col_failed = st.columns(2)
  with col_run:
    run_selected = st.button("Run Selected Jobs", use_container_width=True, disabled=not bool(selected_jobs))
  with col_failed:
    run_failed = st.button(
      "Rerun Failed Jobs",
      use_container_width=True,
      disabled=not any(status == "failed" for status in status_map.values()),
    )

  if run_selected or run_failed:
    if run_failed:
      target_ids = [job_id for job_id, status in status_map.items() if status == "failed"]
    else:
      target_ids = selected_jobs
    target_rows = [row for row in rows if row["job_id"] in set(target_ids)]
    progress = st.progress(0.0)
    for idx, row in enumerate(target_rows, start=1):
      status_map[row["job_id"]] = "running"
      result = run_plan_job(row, log_dir=log_dir)
      status_map[row["job_id"]] = result["status"]
      st.session_state.setdefault("protocol_job_logs", {})[row["job_id"]] = result
      progress.progress(idx / max(1, len(target_rows)))
    st.rerun()

  status_frame = pd.DataFrame(
    [{"job_id": job_id, "status": status} for job_id, status in status_map.items()]
  )
  st.dataframe(status_frame, hide_index=True, use_container_width=True)

  logs = st.session_state.get("protocol_job_logs", {})
  if logs:
    with st.expander("Execution Logs", expanded=False):
      for job_id, result in logs.items():
        st.markdown(f"**{job_id}** - `{result.get('status')}`")
        log_path = Path(result.get("log_path", ""))
        if log_path.exists():
          text = log_path.read_text(encoding="utf-8", errors="replace")
          st.code(text[-8000:], language="text")
        else:
          st.caption(result)


def _render_results_summary(configs: Sequence[Mapping[str, Any]]) -> None:
  if collect_result_rows is None or summarize_results is None:
    return

  st.markdown("#### Results Summary")
  default_root = "results"
  if configs:
    default_root = str(configs[0].get("output_dir", "results"))
  result_root = st.text_input(
    "Results root to scan",
    value=st.session_state.get("protocol_result_root", default_root),
    key="protocol_result_root",
  )

  if st.button("Scan Results", use_container_width=True):
    try:
      results = collect_result_rows(result_root)
      summary = summarize_results(results)
      st.session_state.protocol_results_frame = results
      st.session_state.protocol_summary_frame = summary
    except Exception as exc:
      st.error(f"Could not scan results: {exc}")

  results = st.session_state.get("protocol_results_frame")
  summary = st.session_state.get("protocol_summary_frame")
  if results is not None:
    st.caption(f"Detailed result rows: {len(results)}")
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.download_button(
      "Download detailed CSV",
      data=results.to_csv(index=False),
      file_name="protocol_results_detailed.csv",
      mime="text/csv",
    )
  if summary is not None and not summary.empty:
    st.caption(f"Summary rows: {len(summary)}")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    c_csv, c_latex = st.columns(2)
    with c_csv:
      st.download_button(
        "Download summary CSV",
        data=summary.to_csv(index=False),
        file_name="protocol_results_summary.csv",
        mime="text/csv",
      )
    with c_latex:
      st.download_button(
        "Download summary LaTeX",
        data=summary.to_latex(index=False, float_format="%.4f"),
        file_name="protocol_results_summary.tex",
        mime="text/x-tex",
      )


def render_protocol_workbench(
  configs: list[dict[str, Any]],
  selected_idx: int,
  command_builder: Callable[[Mapping[str, Any]], Sequence[str]],
) -> None:
  """Render validation, sweep, execution and aggregation tools."""

  if not _registry_available():
    return

  with st.expander("Protocol Validation, Sweeps, Execution and Results", expanded=False):
    tabs = st.tabs(["Validate", "Sweeps", "Plan", "Execute", "Results"])
    with tabs[0]:
      _render_validation(configs)
    with tabs[1]:
      _render_sweeps(configs, selected_idx)
    with tabs[2]:
      _render_job_plan(configs, command_builder)
    with tabs[3]:
      _render_execution()
    with tabs[4]:
      _render_results_summary(configs)
