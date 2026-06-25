"""Data loading helpers for Results Explorer."""

from typing import Dict, Optional
import pandas as pd
import streamlit as st

from . import legacy


def load_benchmark_detailed(filepath: str) -> Optional[pd.DataFrame]:
  return legacy.load_benchmark_detailed(filepath)


def load_analysis_results(filepath: str) -> Optional[pd.DataFrame]:
  return legacy.load_analysis_results(filepath)


def load_labels_from_directory(labels_dir: str):
  return legacy.load_labels_from_directory(labels_dir)


def detect_result_type(results_dir: str) -> str:
  return legacy.detect_result_type(results_dir)


def load_results_from_directory(results_dir: str, condition_name: str) -> Optional[Dict]:
  return legacy.load_results_from_directory(results_dir, condition_name)


def aggregate_metrics(all_data: Dict[str, Dict]) -> pd.DataFrame:
  return legacy.aggregate_metrics(all_data)


def is_result_directory(path: str) -> bool:
  return legacy.is_result_directory(path)


@st.cache_data(show_spinner=False)
def scan_for_results(base_path: str, max_depth: int = 5):
  return legacy.scan_for_results(base_path, max_depth=max_depth)


@st.cache_data(show_spinner=False)
def resolve_results_root() -> str:
  return legacy.resolve_results_root()


def select_latest_result_path_per_condition(found_paths, batch_root: str):
  return legacy._select_latest_result_path_per_condition(found_paths, batch_root)
