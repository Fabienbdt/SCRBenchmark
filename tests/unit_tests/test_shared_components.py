"""Tests for shared Streamlit rendering helpers."""

import pandas as pd

from scrbenchmark.gui import shared_components


def test_render_html_table_escapes_dataframe_values(monkeypatch):
  rendered = {}

  def capture_markdown(body, **kwargs):
    rendered["body"] = body
    rendered.update(kwargs)

  monkeypatch.setattr(shared_components.st, "markdown", capture_markdown)
  table = pd.DataFrame({"value": ["<script>alert(1)</script>"]})

  shared_components._render_html_table(table, hide_index=True)

  assert "<script>" not in rendered["body"]
  assert "&lt;script&gt;" in rendered["body"]
  assert rendered["unsafe_allow_html"] is True


def test_render_html_table_accepts_pandas_styler(monkeypatch):
  rendered = {}
  monkeypatch.setattr(
    shared_components.st,
    "markdown",
    lambda body, **kwargs: rendered.update(body=body, **kwargs),
  )
  table = pd.DataFrame({
    "label": ["<script>alert(1)</script>"],
    "count": [1234],
  }).style.format({"count": "{:,}"}, escape="html")

  shared_components._render_html_table(table)

  assert "1,234" in rendered["body"]
  assert "<script>" not in rendered["body"]
  assert "&lt;script&gt;" in rendered["body"]
  assert "<table" in rendered["body"]
