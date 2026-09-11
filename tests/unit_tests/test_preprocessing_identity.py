"""Protect observation identity and raw counts through preprocessing."""

import anndata as ad
import numpy as np
import pytest

from scrbenchmark.utils.data_handler import DataHandler


def test_filtering_preserves_cell_identity_and_selected_annotations(synthetic_data_path):
    data = ad.read_h5ad(synthetic_data_path)
    data.obs["requested"] = ["requested_" + str(i % 3) for i in range(data.n_obs)]
    data.obs["custom_annotation"] = np.arange(data.n_obs)
    data.X[0] = 0
    data.write_h5ad(synthetic_data_path)
    handler = DataHandler().load(synthetic_data_path)
    handler.set_label_column("requested")
    handler.preprocess({"min_genes_per_cell": 1, "min_cells_per_gene": 1, "do_hvg": False})
    processed = handler.get_data()
    assert processed.obs_names.tolist() == data.obs_names[1:].tolist()
    assert processed.obs["requested"].tolist() == data.obs["requested"].iloc[1:].tolist()
    assert processed.obs["labels"].tolist() == data.obs["requested"].iloc[1:].tolist()
    assert processed.obs["custom_annotation"].tolist() == list(range(1, data.n_obs))
    np.testing.assert_array_equal(processed.layers["original_X"], data.X[1:])
    np.testing.assert_array_equal(handler.get_labels(), processed.obs["labels_encoded"].to_numpy())


def test_skipping_preprocessing_keeps_the_selected_labels(synthetic_data_path):
    data = ad.read_h5ad(synthetic_data_path)
    data.obs["requested"] = ["chosen_" + str(i % 2) for i in range(data.n_obs)]
    data.write_h5ad(synthetic_data_path)
    handler = DataHandler().load(synthetic_data_path)
    handler.set_label_column("requested")
    expected = handler.get_labels().copy()
    handler.preprocess({"skip": True})
    np.testing.assert_array_equal(handler.get_labels(), expected)


@pytest.mark.parametrize("params", [
    {"min_genes_per_cell": 10000},
    {"min_genes_per_cell": 1, "min_cells_per_gene": 10000},
])
def test_filters_that_remove_all_data_raise_an_actionable_error(synthetic_data_path, params):
    handler = DataHandler().load(synthetic_data_path)
    with pytest.raises(ValueError, match="Preprocessing removed all"):
        handler.preprocess(params)


def test_import_time_correction_keeps_raw_counts(synthetic_data_path):
    handler = DataHandler().load(synthetic_data_path)
    original = handler.adata.X.copy()
    handler.adata.layers["original_X"] = original.copy()
    corrected = np.log1p(original) - 1
    handler.adata.X = corrected.copy()
    handler.adata.uns["batch_correction"] = {"stage": "import", "method": "test"}
    handler._batch_correction_applied = True
    handler.preprocess({"do_cell_filtering": False, "do_gene_filtering": False,
                        "do_hvg": False, "do_scaling": False})
    np.testing.assert_array_equal(handler.adata.X, corrected)
    np.testing.assert_array_equal(handler.adata.layers["original_X"], original)
    np.testing.assert_array_equal(handler.adata.raw.X, original)
    assert handler.adata.uns["batch_correction"]["stage"] == "import"
