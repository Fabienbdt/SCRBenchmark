"""
Reusable Streamlit widgets and UI helpers.

This module is intentionally UI-only and must not contain scientific logic.
"""

from __future__ import annotations

import io
import traceback
import zipfile
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd
import streamlit as st

from core.config import HyperparameterConfig, ParamType
from gui.constants import BATCH_COLUMN_CANDIDATES, LABEL_COLUMN_CANDIDATES
from gui.i18n import t


def check_prerequisites(
  require_data: bool = False,
  require_preprocessing: bool = False,
  require_algorithms: bool = False,
  require_results: bool = False,
) -> bool:
  """Check common page prerequisites with consistent user-facing warnings."""
  if require_data:
    handler = st.session_state.get("data_handler")
    if handler is None:
      st.warning(t("warnings.load_data_first"))
      return False

  if require_preprocessing and not st.session_state.get("data_preprocessed", False):
    st.warning(t("warnings.preprocess_first"))
    return False

  if require_algorithms and not st.session_state.get("selected_algorithms", []):
    st.warning(t("warnings.select_algorithms_first"))
    return False

  if require_results:
    has_results = any(
      st.session_state.get(key)
      for key in ("analysis_results", "benchmark_results", "explorer_results")
    )
    if not has_results:
      st.warning(t("warnings.results_missing"))
      return False

  return True


def _is_int_like(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def render_synced_number_input(
  label: str,
  key: str,
  min_value: float | int,
  max_value: float | int,
  default: float | int,
  step: float | int,
  help_text: str = "",
  col_ratios: Sequence[int] = (3, 1),
) -> float | int:
  """
  Render synchronized slider + number_input backed by a single session key.
  """
  # Clamp default and any existing session value to the valid range
  # (prevents stale values from a previous dataset exceeding new limits)
  def _clamp_init(v):
    return max(min_value, min(max_value, v))

  if key not in st.session_state:
    st.session_state[key] = _clamp_init(default)
  else:
    st.session_state[key] = _clamp_init(st.session_state[key])

  slider_key = f"{key}__slider"
  input_key = f"{key}__input"
  if slider_key not in st.session_state:
    st.session_state[slider_key] = st.session_state[key]
  else:
    st.session_state[slider_key] = _clamp_init(st.session_state[slider_key])
  if input_key not in st.session_state:
    st.session_state[input_key] = st.session_state[key]
  else:
    st.session_state[input_key] = _clamp_init(st.session_state[input_key])

  is_int_mode = all(
    _is_int_like(v) for v in (min_value, max_value, default, step)
  )

  def _cast(value: Any) -> float | int:
    if is_int_mode:
      return int(value)
    return float(value)

  def _clamp(value: float | int) -> float | int:
    clamped = max(min_value, min(max_value, value))
    return _cast(clamped)

  def _from_slider() -> None:
    value = _clamp(st.session_state[slider_key])
    st.session_state[key] = value
    st.session_state[input_key] = value

  def _from_input() -> None:
    value = _clamp(st.session_state[input_key])
    st.session_state[key] = value
    st.session_state[slider_key] = value

  col_slider, col_input = st.columns(col_ratios)
  with col_slider:
    st.slider(
      label,
      min_value=_cast(min_value),
      max_value=_cast(max_value),
      value=_cast(st.session_state[slider_key]),
      step=_cast(step),
      key=slider_key,
      on_change=_from_slider,
      label_visibility="collapsed",
      help=help_text,
    )
  with col_input:
    st.number_input(
      label,
      min_value=_cast(min_value),
      max_value=_cast(max_value),
      value=_cast(st.session_state[input_key]),
      step=_cast(step),
      key=input_key,
      on_change=_from_input,
      label_visibility="visible",
      help=help_text,
    )

  return _cast(st.session_state[key])


def _param_type_name(hp: HyperparameterConfig) -> str:
  raw = getattr(hp, "param_type", None)
  if isinstance(raw, ParamType):
    return raw.value
  return str(raw).lower()


def _safe_int(value: Any, default: int) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return int(default)


def _safe_float(value: Any, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return float(default)


def _render_bounded_float_precision(
  hp: HyperparameterConfig,
  current_value: Any,
  key: str,
  help_text: str,
) -> float:
  """
  Render bounded float with slider + unbounded precise number input.

  The slider stays clamped to [min, max] while the number input keeps full precision.
  """
  if hp.min_value is None or hp.max_value is None:
    default_value = _safe_float(hp.default, 0.0) if hp.default is not None else 0.0
    return _safe_float(current_value, default_value)

  value_key = f"{key}__value"
  slider_key = f"{key}__slider"
  input_key = f"{key}__input"

  slider_min = float(hp.min_value)
  slider_max = float(hp.max_value)
  step = float(hp.step) if hp.step is not None else 0.0001
  default_value = _safe_float(hp.default, slider_min) if hp.default is not None else slider_min
  current = _safe_float(current_value, default_value)

  if value_key not in st.session_state:
    st.session_state[value_key] = current
  if slider_key not in st.session_state:
    clamped = max(slider_min, min(slider_max, float(st.session_state[value_key])))
    st.session_state[slider_key] = clamped
  if input_key not in st.session_state:
    st.session_state[input_key] = float(st.session_state[value_key])

  def _from_slider() -> None:
    val = float(st.session_state[slider_key])
    st.session_state[value_key] = val
    st.session_state[input_key] = val

  def _from_input() -> None:
    val = float(st.session_state[input_key])
    st.session_state[value_key] = val
    clamped = max(slider_min, min(slider_max, val))
    st.session_state[slider_key] = clamped

  st.markdown(f"**{hp.display_name}**")
  if hp.description:
    st.caption(hp.description)

  col_slider, col_input = st.columns([3, 1])
  with col_slider:
    st.slider(
      "Slider",
      min_value=slider_min,
      max_value=slider_max,
      value=float(st.session_state[slider_key]),
      step=step,
      format="%.4f",
      key=slider_key,
      label_visibility="collapsed",
      on_change=_from_slider,
      help=help_text,
    )
  with col_input:
    st.number_input(
      "Value",
      value=float(st.session_state[input_key]),
      step=step,
      format="%.4f",
      key=input_key,
      label_visibility="collapsed",
      on_change=_from_input,
      help=help_text,
    )

  final_value = float(st.session_state[value_key])
  if final_value < slider_min or final_value > slider_max:
    st.info(
      f"Value {final_value:.4f} exceeds slider range [{slider_min}, {slider_max}]. "
      "Using precise value."
    )
  return final_value


def render_param_input(
  hp: HyperparameterConfig,
  current_value: Any,
  key_prefix: str,
  compact: bool = False,
  bounded_float_mode: str = "default",
) -> Any:
  """
  Render a robust parameter input widget for algorithm hyperparameters.
  """
  key = f"{key_prefix}_{hp.name}"
  param_type = _param_type_name(hp)
  help_text = (hp.description or "").strip()
  if getattr(hp, "tuning_guide", None):
    help_text = f"{help_text}\n\n{hp.tuning_guide}".strip()

  if param_type == ParamType.INTEGER.value:
    default_value = _safe_int(hp.default, 0) if hp.default is not None else 0
    return st.number_input(
      hp.display_name,
      min_value=int(hp.min_value) if hp.min_value is not None else None,
      max_value=int(hp.max_value) if hp.max_value is not None else None,
      value=_safe_int(current_value, default_value),
      step=int(hp.step) if hp.step is not None else 1,
      key=key,
      help=help_text,
    )

  if param_type == ParamType.FLOAT.value:
    if (
      bounded_float_mode == "precision_slider"
      and hp.min_value is not None
      and hp.max_value is not None
    ):
      return _render_bounded_float_precision(
        hp=hp,
        current_value=current_value,
        key=key,
        help_text=help_text,
      )

    default_value = _safe_float(hp.default, 0.0) if hp.default is not None else 0.0
    if (
      not compact
      and hp.min_value is not None
      and hp.max_value is not None
      and hp.step is not None
    ):
      return render_synced_number_input(
        label=hp.display_name,
        key=key,
        min_value=float(hp.min_value),
        max_value=float(hp.max_value),
        default=_safe_float(current_value, default_value),
        step=float(hp.step),
        help_text=help_text,
      )
    return st.number_input(
      hp.display_name,
      min_value=float(hp.min_value) if hp.min_value is not None else None,
      max_value=float(hp.max_value) if hp.max_value is not None else None,
      value=_safe_float(current_value, default_value),
      step=float(hp.step) if hp.step is not None else 0.001,
      format="%.6f",
      key=key,
      help=help_text,
    )

  if param_type == ParamType.BOOLEAN.value:
    return st.checkbox(
      hp.display_name,
      value=bool(current_value),
      key=key,
      help=help_text,
    )

  if param_type == ParamType.CHOICE.value:
    choices = list(hp.choices or [])
    if not choices:
      return st.text_input(hp.display_name, value=str(current_value), key=key, help=help_text)
    idx = choices.index(current_value) if current_value in choices else 0
    return st.selectbox(hp.display_name, options=choices, index=idx, key=key, help=help_text)

  if param_type == ParamType.MULTI_CHOICE.value:
    choices = list(hp.choices or [])
    if isinstance(current_value, list):
      default = [x for x in current_value if x in choices]
    elif current_value in choices:
      default = [current_value]
    else:
      default = []
    return st.multiselect(hp.display_name, options=choices, default=default, key=key, help=help_text)

  if param_type == ParamType.RANGE.value:
    min_v = _safe_float(hp.min_value, 0.0) if hp.min_value is not None else 0.0
    max_v = _safe_float(hp.max_value, 1.0) if hp.max_value is not None else 1.0
    if min_v > max_v:
      min_v, max_v = max_v, min_v

    if isinstance(current_value, (list, tuple)) and len(current_value) == 2:
      low = _safe_float(current_value[0], min_v)
      high = _safe_float(current_value[1], max_v)
    else:
      low = _safe_float(hp.default[0], min_v) if isinstance(hp.default, (list, tuple)) and len(hp.default) == 2 else min_v
      high = _safe_float(hp.default[1], max_v) if isinstance(hp.default, (list, tuple)) and len(hp.default) == 2 else max_v

    low = max(min_v, min(max_v, low))
    high = max(min_v, min(max_v, high))
    if low > high:
      low, high = high, low

    range_step = _safe_float(hp.step, 0.001) if hp.step is not None else None
    low_sel, high_sel = st.slider(
      hp.display_name,
      min_value=float(min_v),
      max_value=float(max_v),
      value=(float(low), float(high)),
      step=range_step,
      key=key,
      help=help_text,
    )
    return [float(low_sel), float(high_sel)]

  return st.text_input(hp.display_name, value=str(current_value), key=key, help=help_text)


def download_button(
  data: Any,
  filename: str,
  label: str,
  mime: Optional[str] = None,
  key: Optional[str] = None,
) -> bool:
  """Smart download button with basic MIME/data-type detection."""
  payload = data
  detected_mime = mime

  if isinstance(data, pd.DataFrame):
    payload = data.to_csv(index=False).encode("utf-8")
    detected_mime = detected_mime or "text/csv"
  elif isinstance(data, str):
    payload = data.encode("utf-8")
    detected_mime = detected_mime or "text/plain"
  elif isinstance(data, bytes):
    detected_mime = detected_mime or "application/octet-stream"
  else:
    payload = str(data).encode("utf-8")
    detected_mime = detected_mime or "text/plain"

  return st.download_button(
    label=label,
    data=payload,
    file_name=filename,
    mime=detected_mime,
    key=key,
  )


def detect_batch_column(adata: Any) -> Optional[str]:
  """Detect a likely batch column from AnnData.obs."""
  if adata is None or not hasattr(adata, "obs"):
    return None
  for column in BATCH_COLUMN_CANDIDATES:
    if column in adata.obs.columns:
      return column
  return None


def detect_label_column(adata: Any, handler: Any = None) -> Optional[str]:
  """Detect a likely label column from AnnData.obs."""
  if adata is None or not hasattr(adata, "obs"):
    return None

  if handler is not None:
    label_key = getattr(handler, "label_key", None)
    if label_key and label_key in adata.obs.columns:
      return label_key

  for column in LABEL_COLUMN_CANDIDATES:
    if column in adata.obs.columns:
      return column
  return None


def display_error(
  error: Exception,
  user_message: str,
  show_traceback: bool = True,
  show_retry: bool = False,
  retry_callback: Optional[Callable[[], None]] = None,
  show_reset: bool = False,
  reset_callback: Optional[Callable[[], None]] = None,
  key_prefix: str = "ui_error",
) -> None:
  """Display user-friendly error with optional technical details and actions."""
  st.error(user_message)

  if show_traceback:
    with st.expander(t("details.technical"), expanded=False):
      st.code(traceback.format_exc())

  if not show_retry and not show_reset:
    return

  cols = st.columns(2)
  if show_retry:
    if cols[0].button(t("actions.retry"), key=f"{key_prefix}_retry"):
      if retry_callback is not None:
        retry_callback()
      else:
        st.rerun()
  if show_reset:
    if cols[1].button(t("actions.reset"), key=f"{key_prefix}_reset"):
      if reset_callback is not None:
        reset_callback()
      else:
        st.rerun()


def _export_figure_bytes(fig: Any, fmt: str, dpi: int) -> bytes:
  buffer = io.BytesIO()
  fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
  buffer.seek(0)
  return buffer.getvalue()


def render_export_panel(
  fig: Any,
  base_filename: str,
  formats: Optional[Iterable[str]] = None,
  show_dpi: bool = True,
) -> None:
  """Render an export panel supporting single format export and ZIP-all."""
  if fig is None:
    st.warning("No figure available to export.")
    return

  available_formats = [str(f).lower() for f in (formats or ["PNG", "SVG", "PDF"])]
  key_prefix = f"export_{base_filename}".replace(" ", "_").replace("/", "_")

  col_a, col_b, col_c = st.columns([2, 1, 1])
  with col_a:
    export_fmt = st.selectbox(
      "Format",
      options=available_formats,
      index=0,
      key=f"{key_prefix}_format",
    )
  with col_b:
    if show_dpi:
      dpi = st.selectbox(
        "DPI",
        options=[150, 300, 600, 1200],
        index=1,
        key=f"{key_prefix}_dpi",
      )
    else:
      dpi = 300

  single_name = f"{base_filename}.{export_fmt}"
  single_payload = _export_figure_bytes(fig, export_fmt, int(dpi))
  with col_c:
    download_button(
      data=single_payload,
      filename=single_name,
      label="Export",
      mime="application/octet-stream",
      key=f"{key_prefix}_single_download",
    )

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for fmt in available_formats:
      name = f"{base_filename}.{fmt}"
      archive.writestr(name, _export_figure_bytes(fig, fmt, int(dpi)))
  zip_buffer.seek(0)

  download_button(
    data=zip_buffer.getvalue(),
    filename=f"{base_filename}_all_formats.zip",
    label="Export All (ZIP)",
    mime="application/zip",
    key=f"{key_prefix}_zip_download",
  )


def _render_layer_list_editor(
  title: str,
  state_key: str,
  default_values: Sequence[int],
  min_size: int,
  max_size: int,
  step: int,
) -> list[int]:
  if state_key not in st.session_state:
    st.session_state[state_key] = [int(v) for v in default_values]
  values = [int(v) for v in st.session_state.get(state_key, [])]

  st.markdown(f"**{title}**")
  add_col, del_col = st.columns(2)
  if add_col.button("+ Layer", key=f"{state_key}_add"):
    values.append(int(default_values[-1] if default_values else max(min_size, 128)))
  if del_col.button("- Last", key=f"{state_key}_del", disabled=len(values) == 0):
    values = values[:-1]

  remove_indices = set()
  updated_values: list[int] = []
  for idx, layer in enumerate(values):
    row_left, row_right = st.columns([5, 1])
    with row_left:
      updated = render_synced_number_input(
        label=f"{title} #{idx + 1}",
        key=f"{state_key}_{idx}",
        min_value=min_size,
        max_value=max_size,
        default=int(layer),
        step=step,
        help_text=f"Size of layer {idx + 1}",
        col_ratios=(3, 1),
      )
      updated_values.append(int(updated))
    with row_right:
      st.markdown("<br>", unsafe_allow_html=True)
      if st.button("Delete", key=f"{state_key}_rm_{idx}"):
        remove_indices.add(idx)

  final_values = [v for i, v in enumerate(updated_values) if i not in remove_indices]
  st.session_state[state_key] = final_values
  return final_values


def _render_architecture_preview(
  encoder_layers: Sequence[int],
  latent_dim: int,
  decoder_layers: Sequence[int],
) -> None:
  import matplotlib.pyplot as plt

  nodes = [*encoder_layers, int(latent_dim), *decoder_layers]
  if not nodes:
    return

  colors = (
    ["#4C78A8"] * len(encoder_layers)
    + ["#E45756"]
    + ["#59A14F"] * len(decoder_layers)
  )
  labels = (
    [f"E{i+1}" for i in range(len(encoder_layers))]
    + ["Z"]
    + [f"D{i+1}" for i in range(len(decoder_layers))]
  )

  fig_w = max(7, 1.2 * len(nodes))
  fig, ax = plt.subplots(figsize=(fig_w, 2.8))
  x = list(range(len(nodes)))
  ax.bar(x, nodes, color=colors, alpha=0.85)
  for i, (xi, yi) in enumerate(zip(x, nodes)):
    ax.text(xi, yi + max(5, 0.02 * max(nodes)), f"{labels[i]}\n{yi}", ha="center", va="bottom", fontsize=8)
  ax.set_xticks([])
  ax.set_ylabel("Units")
  ax.set_title("Architecture preview: Encoder (blue) → Latent (red) → Decoder (green)", fontsize=10)
  ax.grid(axis="y", alpha=0.25)
  st.pyplot(fig)
  plt.close(fig)


def render_architecture_editor(
  algo_name: str,
  *,
  encoder_default: Sequence[int],
  latent_default: int,
  decoder_default: Sequence[int],
  decoder_note: Optional[str] = None,
  parse_layers: Optional[Callable[[str], list[int]]] = None,
  format_layers: Optional[Callable[[Sequence[int]], str]] = None,
  min_size: int = 16,
  max_size: int = 2048,
  step: int = 16,
) -> tuple[list[int], list[int], list[int]]:
  """
  Visual editor for network architecture with optional advanced text mode.
  """
  prefix = f"{algo_name}_arch_editor"
  toggle_widget = getattr(st, "toggle", None)
  if callable(toggle_widget):
    show_advanced = toggle_widget(
      "Advanced mode (text input)",
      value=st.session_state.get(f"{prefix}_advanced", False),
      key=f"{prefix}_advanced",
    )
  else:
    show_advanced = st.checkbox(
      "Advanced mode (text input)",
      value=st.session_state.get(f"{prefix}_advanced", False),
      key=f"{prefix}_advanced",
    )

  if decoder_note:
    st.info(decoder_note)

  if show_advanced:
    if parse_layers is None:
      parse_layers = lambda raw: [int(x.strip()) for x in str(raw).replace("[", "").replace("]", "").split(",") if x.strip()]
    if format_layers is None:
      format_layers = lambda values: "[" + ",".join(str(int(v)) for v in values) + "]"

    enc_txt = st.text_input(
      "Encoder layers",
      value=format_layers(encoder_default),
      key=f"{prefix}_enc_txt",
      help="Exemple: [256,64]",
    )
    lat_txt = st.text_input(
      "Latent space",
      value=format_layers([int(latent_default)]),
      key=f"{prefix}_lat_txt",
      help="Exemple: [32]",
    )
    dec_txt = st.text_input(
      "Decoder layers",
      value=format_layers(decoder_default),
      key=f"{prefix}_dec_txt",
      help="Exemple: [64,256]",
    )
    encoder_layers = parse_layers(enc_txt)
    latent_layers = parse_layers(lat_txt) or [int(latent_default)]
    decoder_layers = parse_layers(dec_txt)
  else:
    st.caption("Add/remove layers and tune their size using slider + numeric input.")
    encoder_layers = _render_layer_list_editor(
      title="Encoder",
      state_key=f"{prefix}_encoder",
      default_values=encoder_default,
      min_size=min_size,
      max_size=max_size,
      step=step,
    )
    decoder_layers = _render_layer_list_editor(
      title="Decoder",
      state_key=f"{prefix}_decoder",
      default_values=decoder_default,
      min_size=min_size,
      max_size=max_size,
      step=step,
    )
    latent_value = render_synced_number_input(
      label="Dimension latente",
      key=f"{prefix}_latent",
      min_value=min_size,
      max_value=max_size,
      default=int(latent_default),
      step=step,
      help_text="Size of the latent space",
      col_ratios=(3, 1),
    )
    latent_layers = [int(latent_value)]

  if latent_layers and len(latent_layers) > 1:
    st.warning("Only one latent dimension is used. The first value will be kept.")
    latent_layers = [latent_layers[0]]

  _render_architecture_preview(
    encoder_layers=encoder_layers,
    latent_dim=int(latent_layers[0]) if latent_layers else int(latent_default),
    decoder_layers=decoder_layers,
  )
  return encoder_layers, latent_layers, decoder_layers
