import csv
import sys
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_ROOT = PROJECT_ROOT / "scripts" / "setup"
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

import download_report_sources  # noqa: E402


REPORT_DATASETS = {
    "bbag094_zeisel",
    "bbag094_spleen",
    "baron_human_pancreas",
    "gse112013_human_testis_raw_counts",
    "kang_pbmc_gse96583_singlets_raw_counts",
    "macaque_retina_gse118480_bipolar_raw_counts",
    "paul15_bone_marrow_raw_counts",
    "Tabula_Muris_liver_filtered_raw_counts",
    "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2",
    "Mouse_Pancreas_1_raw_counts",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_source_manifest_covers_every_report_dataset():
    rows = download_report_sources.read_manifest(
        PROJECT_ROOT / "data" / "report_sources" / "source_manifest.csv"
    )
    covered = set().union(
        *(download_report_sources.split_values(row["dataset_keys"]) for row in rows)
    )

    assert REPORT_DATASETS <= covered
    assert len({row["asset_id"] for row in rows}) == len(rows)
    assert all(row["url"].startswith("https://") for row in rows)
    assert all(row["size_bytes"].isdigit() and int(row["size_bytes"]) > 0 for row in rows)


def test_source_selection_rejects_an_unknown_dataset():
    rows = download_report_sources.read_manifest(
        PROJECT_ROOT / "data" / "report_sources" / "source_manifest.csv"
    )
    args = Namespace(all=False, datasets="does_not_exist", assets=None)

    try:
        download_report_sources.select_rows(rows, args)
    except ValueError as error:
        assert "does_not_exist" in str(error)
    else:
        raise AssertionError("Unknown dataset selection should fail")


def test_report_pancreas_alias_and_dataset_metadata_are_reconciled():
    report_rows = _read_csv(
        PROJECT_ROOT / "reproducibility" / "report_reproduction_map.csv"
    )
    holdout = next(
        row for row in report_rows if row["report_label"] == "tab:scraw_holdout_pancreas_results"
    )
    assert "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2" in holdout[
        "datasets"
    ]
    assert "Human_Pancreas_1_raw_counts" not in holdout["datasets"]

    prepared_rows = _read_csv(
        PROJECT_ROOT / "data" / "stable_generalist" / "download_manifest.csv"
    )
    prepared = {row["dataset_key"]: row for row in prepared_rows}
    assert prepared["pancreas_raw_counts"]["h5ad_n_obs"] == "14908"
    assert prepared[
        "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2"
    ]["h5ad_n_obs"] == "6339"

    table_rows = _read_csv(
        PROJECT_ROOT
        / "reproducibility"
        / "stable_generalist"
        / "stable_generalist_dataset_table.csv"
    )
    table = {row["dataset_key"]: row for row in table_rows}
    assert table["pancreas_raw_counts"]["n_cells"] == "14908"


def test_tabula_source_is_the_exact_dimension_senis_object():
    rows = _read_csv(PROJECT_ROOT / "data" / "report_sources" / "source_manifest.csv")
    liver = next(row for row in rows if row["asset_id"] == "tabula_muris_senis_liver_h5ad")

    assert liver["size_bytes"] == "134053262"
    assert liver["checksum_algorithm"] == "md5"
    assert liver["checksum"] == "05cb7669c7439562faa7170dc6896dce"
    assert liver["url"].endswith("/23872526")


def test_maintainer_facing_metadata_and_protocol_names_are_in_english():
    forbidden = {
        "Rapport -",
        "donneur / individu",
        "pas de batch explicite",
        "Aucun batch explicite",
        "Faible",
        "Fort",
        "Intermédiaire",
        "aucune",
        "echantillon",
        "sequencage",
        "imbrique",
        "condition biologique",
        "poids faible",
        "Poids de reconstruction",
        "jaune vif",
        "opacite",
    }
    checked_files = [
        PROJECT_ROOT / "src" / "scrbenchmark" / "gui" / "customize_benchmark.py",
        PROJECT_ROOT
        / "vendor"
        / "scraw_dedicated"
        / "src"
        / "scraw_dedicated"
        / "visualization.py",
        *sorted((PROJECT_ROOT / "protocols" / "report").glob("*.yaml")),
        PROJECT_ROOT
        / "reproducibility"
        / "stable_generalist"
        / "stable_generalist_dataset_table.csv",
        PROJECT_ROOT
        / "scraw-transductive-stable-generalist"
        / "metadata"
        / "stable_generalist_dataset_table.csv",
    ]

    for path in checked_files:
        content = path.read_text(encoding="utf-8")
        remaining = sorted(term for term in forbidden if term in content)
        assert not remaining, (path, remaining)
