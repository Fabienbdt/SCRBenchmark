"""Render public Streamlit pages without a browser or network dataset access."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import scrbenchmark


@pytest.mark.parametrize("page", [
    "Data Upload", "Documentation", "Customize Benchmark", "Report Reproduction", "Results Explorer",
])
def test_gui_page_renders_without_exception(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(Path(scrbenchmark.__file__).with_name("app.py"), default_timeout=60)
    app.session_state["current_page"] = page
    app.run()
    assert not app.exception, [(error.message, error.stack_trace) for error in app.exception]
    assert app.sidebar.button


@pytest.mark.parametrize("page", ["Data Split", "Preprocessing", "Algorithm Config", "Analysis"])
def test_workflow_pages_render_with_loaded_data(page, toy_h5ad, tmp_path, monkeypatch):
    from scrbenchmark.utils.data_handler import DataHandler
    handler = DataHandler()
    handler.load(str(toy_h5ad))
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(Path(scrbenchmark.__file__).with_name("app.py"), default_timeout=60)
    app.session_state["current_page"] = page
    app.session_state["data_handler"] = handler
    app.session_state["data_loaded"] = True
    app.session_state["uploaded_file_path"] = str(toy_h5ad)
    app.session_state["selected_algorithms"] = ["pca"]
    app.session_state["algorithm_params"] = {"pca": {"clustering_method": "kmeans", "n_clusters": 3}}
    app.run()
    assert not app.exception, [(error.message, error.stack_trace) for error in app.exception]
