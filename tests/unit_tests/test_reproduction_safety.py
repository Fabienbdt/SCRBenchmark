import csv
import sys
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = PROJECT_ROOT / "scripts" / "reproduction"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (REPRODUCTION_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_stable_generalist_plan  # noqa: E402
import download_datasets  # noqa: E402
import export_existing_scraw_artifacts  # noqa: E402
import prepare_stable_generalist_data  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_verify_only_does_not_write_to_the_dataset_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "download_manifest.csv"
    table = tmp_path / "dataset_table.csv"
    target_root = tmp_path / "missing_datasets"
    manifest_row = {
        "dataset_key": "demo",
        "dataset": "Demo",
        "filename": "demo.h5ad",
        "size_bytes": "1",
        "sha256": "0" * 64,
        "h5ad_n_obs": "1",
        "h5ad_n_vars": "1",
        "n_labels": "1",
        "label_key": "label",
        "batch_key": "batch",
        "source_url": "",
    }
    _write_csv(manifest, list(manifest_row), [manifest_row])
    table_row = {
        "dataset_key": "demo",
        "data_file": "demo.h5ad",
        "n_labels": "1",
        "label_key": "label",
        "dann_batch_column": "batch",
    }
    _write_csv(table, list(table_row), [table_row])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_datasets.py",
            "--manifest",
            str(manifest),
            "--dataset-table",
            str(table),
            "--target-root",
            str(target_root),
            "--verify-only",
            "--skip-reference-table-check",
        ],
    )

    assert download_datasets.main() == 1
    assert not target_root.exists()
    assert manifest.read_text(encoding="utf-8").startswith("dataset_key,dataset,filename")


def test_export_dry_run_does_not_create_an_output_tree(tmp_path, monkeypatch):
    manifest = tmp_path / "download_manifest.csv"
    _write_csv(manifest, ["dataset_key"], [{"dataset_key": "demo"}])
    output_root = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_existing_scraw_artifacts.py",
            "--manifest",
            str(manifest),
            "--weights-root",
            str(tmp_path / "weights"),
            "--model-root",
            str(tmp_path / "models"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert export_existing_scraw_artifacts.main() == 0
    assert not output_root.exists()


def test_legacy_prepare_dry_run_does_not_create_an_output_tree(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "demo.h5ad"
    source_file.write_bytes(b"demo")
    dataset_table = tmp_path / "datasets.csv"
    _write_csv(
        dataset_table,
        ["dataset_key", "dataset", "data_file"],
        [{"dataset_key": "demo", "dataset": "Demo", "data_file": "demo.h5ad"}],
    )
    target_root = tmp_path / "target"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stable_generalist_data.py",
            "--dataset-table",
            str(dataset_table),
            "--source-root",
            str(source_root),
            "--target-root",
            str(target_root),
            "--dry-run",
        ],
    )

    assert prepare_stable_generalist_data.main() == 0
    assert not target_root.exists()


def test_stable_plan_blocks_missing_data_by_default(tmp_path):
    dataset_table = tmp_path / "datasets.csv"
    results_table = tmp_path / "results.csv"
    data_root = tmp_path / "data"
    _write_csv(
        dataset_table,
        ["dataset_key", "dataset", "data_file", "label_key", "dann_batch_column", "n_labels"],
        [
            {
                "dataset_key": "demo",
                "dataset": "Demo",
                "data_file": "demo.h5ad",
                "label_key": "Group",
                "dann_batch_column": "batch",
                "n_labels": 3,
            }
        ],
    )
    _write_csv(
        results_table,
        ["result_row_id", "dataset_key", "dataset", "method", "n_clusters_found"],
        [
            {
                "result_row_id": "demo_pca",
                "dataset_key": "demo",
                "dataset": "Demo",
                "method": "PCA",
                "n_clusters_found": 3,
            }
        ],
    )
    args = Namespace(
        results_table=str(results_table),
        dataset_table=str(dataset_table),
        data_root=str(data_root),
        output_root=str(tmp_path / "out"),
        python_bin="python3",
        device="cpu",
        seed=42,
        scib_n_jobs=1,
        datasets="",
        methods="",
        allow_missing_data=False,
        strict_data=False,
    )

    rows = build_stable_generalist_plan.build_plan(args)

    assert len(rows) == 1
    assert rows[0]["status"] == "blocked_missing_data"
    assert "Missing data file" in rows[0]["notes"]
