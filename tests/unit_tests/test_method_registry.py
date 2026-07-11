import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reproduction"))

from scrbenchmark.methods import get_method_spec, load_method_specs  # noqa: E402
from scrbenchmark.methods.registry import MethodSpec  # noqa: E402
from run_method import build_command, finalize_method_output  # noqa: E402
from manual_protocols import ManualProtocolConfig, build_jobs  # noqa: E402


def test_report_methods_are_registered():
    specs = load_method_specs()
    canonical = {spec.name for spec in specs.values()}
    expected = {
        "scRAW",
        "scMAE",
        "scNAME",
        "DESC",
        "DESC_scRAW_weighted",
        "DeepScena",
        "CellSIUS",
        "GiniClust",
        "scAIDE",
        "scCAD",
        "Harmony",
        "ComBat",
        "ComBat-seq",
        "Scanorama",
        "PCA",
        "kmeans",
        "louvain",
        "leiden",
        "hdbscan",
        "pca_leiden",
        "scvi",
        "scCDCG",
    }
    assert expected.issubset(canonical)


def test_alias_lookup_returns_canonical_spec():
    assert get_method_spec("scraw").name == "scRAW"
    assert get_method_spec("sc_mae").name == "scMAE"
    assert get_method_spec("combat_seq").name == "ComBat-seq"
    assert get_method_spec("K-Means").name == "kmeans"
    assert get_method_spec("HDBSCAN").name == "hdbscan"


def test_method_specs_declare_core_status_and_runner():
    specs = [spec for spec in load_method_specs().values() if spec.name == "DESC"]
    assert specs
    desc = specs[0]
    assert desc.runner_kind == "command_template"
    assert desc.core_contract == "author_core_unchanged"
    assert desc.core_status


def test_desc_weighted_is_explicit_experimental_variant():
    spec = get_method_spec("desc_scraw_weighted")
    assert spec.name == "DESC_scRAW_weighted"
    assert spec.runner_kind == "command_template"
    assert spec.family == "loss_transfer"
    assert spec.core_contract == "experimental_plugin_explicitly_not_author_baseline"


def _runner_args(tmp_path: Path):
    import argparse

    return argparse.Namespace(
        method="Toy",
        data=str(tmp_path / "toy.h5ad"),
        output=str(tmp_path / "out"),
        dataset_key="toy",
        label_key="Group",
        batch_key="batch",
        n_labels=3,
        seed=7,
        device="cpu",
        scib_n_jobs=1,
        python_bin="python3",
        n_top_genes=2000,
        min_genes_per_cell=200,
        max_genes_per_cell=10000,
        min_cells_per_gene=3,
        target_sum=20000.0,
        scale_max_value=10.0,
        hvg_flavor="seurat",
        n_pcs=50,
        harmony_max_iter=10,
        harmony_nclust=50,
        resolutions="0.5,1.0",
        selection_expected_n_classes=0,
        param=["Toy:alpha=1"],
        overwrite=False,
        verbose=True,
        dry_run=True,
    )


def test_command_template_runner_expands_placeholders(tmp_path):
    spec = MethodSpec.from_mapping(
        {
            "name": "Toy",
            "display_name": "Toy",
            "family": "external",
            "runner": {
                "kind": "command_template",
                "command": [
                    "{python_bin}",
                    "{source_path}/main.py",
                    "--data",
                    "{data}",
                    "--output",
                    "{output}",
                    "--k",
                    "{n_labels}",
                    "{verbose_flag}",
                    "{param_args}",
                ],
            },
            "source": {"kind": "original_source", "path": "external/original_code/toy"},
            "core_contract": "author_core_unchanged",
            "core_status": "test",
        }
    )

    command = build_command(spec, _runner_args(tmp_path))

    assert command[:2] == ["python3", str(PROJECT_ROOT / "external/original_code/toy/main.py")]
    assert "--k" in command
    assert "3" in command
    assert "--verbose" in command
    assert command[-2:] == ["--param", "Toy:alpha=1"]


def test_scraw_command_template_uses_public_preset(tmp_path):
    spec = get_method_spec("scRAW")
    args = _runner_args(tmp_path)
    args.method = "scRAW"
    args.scraw_preset = "baron"

    command = build_command(spec, args)

    assert "--preset" in command
    assert command[command.index("--preset") + 1] == "baron"


def test_scraw_uses_the_vendored_source():
    spec = get_method_spec("scRAW")
    source_path = PROJECT_ROOT / str(spec.source["path"])

    assert spec.source["kind"] == "vendored_source"
    assert source_path == PROJECT_ROOT / "vendor" / "scraw_inductive"
    assert (source_path / "src" / "scraw" / "pipeline.py").exists()


def test_command_template_can_normalize_labels_file(tmp_path):
    import pandas as pd

    labels_path = tmp_path / "out" / "raw" / "labels.csv"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text("cell_id,cluster\nc1,A\nc2,B\n", encoding="utf-8")

    spec = MethodSpec.from_mapping(
        {
            "name": "Toy",
            "display_name": "Toy",
            "family": "external",
            "runner": {"kind": "command_template", "command": ["true"]},
            "source": {"kind": "original_source", "path": "external/original_code/toy"},
            "core_contract": "author_core_unchanged",
            "core_status": "test",
            "output": {
                "expected_file": "results/analysis_results.csv",
                "labels_file": "raw/labels.csv",
                "labels_column": "cluster",
                "cell_id_column": "cell_id",
            },
        }
    )

    finalize_method_output(spec, _runner_args(tmp_path))

    result_path = tmp_path / "out" / "results" / "analysis_results.csv"
    result = pd.read_csv(result_path)
    assert list(result.columns) == ["cell_id", "cluster"]
    assert result.to_dict(orient="records") == [
        {"cell_id": "c1", "cluster": "A"},
        {"cell_id": "c2", "cluster": "B"},
    ]


def test_manual_loss_transfer_protocol_builds_weighted_jobs(tmp_path):
    config = ManualProtocolConfig(
        protocol="loss_transfer",
        data=str(tmp_path / "toy.h5ad"),
        output_root=str(tmp_path / "out"),
        dataset_key="toy",
        label_key="Group",
        batch_key="batch",
        n_labels=3,
        seeds=(42,),
        device="cpu",
        python_bin="python3",
        loss_methods=("scDeepCluster",),
        loss_variants=("weighted", "triplet"),
        loss_weight_params={"warmup_epochs": "12"},
    )

    commands = [row["command"] for row in build_jobs(config)]

    assert len(commands) == 2
    assert "--method scDeepCluster_scRAW_weighted" in commands[0]
    assert "--param scdeepcluster_scraw_weighted:warmup_epochs=12" in commands[0]
    assert "--param scdeepcluster_scraw_weighted:rare_triplet_weight=0.05" in commands[1]


def test_manual_harmony_protocol_builds_custom_harmony_command(tmp_path):
    config = ManualProtocolConfig(
        protocol="harmony",
        data=str(tmp_path / "toy.h5ad"),
        output_root=str(tmp_path / "out"),
        dataset_key="toy",
        label_key="Group",
        batch_key="batch",
        n_labels=3,
        seeds=(7,),
        device="cpu",
        python_bin="python3",
        harmony_methods=("scMAE+Harmony",),
        n_pcs=20,
        harmony_max_iter=3,
        harmony_nclust=12,
        params=("sc_mae:epochs=2",),
    )

    command = build_jobs(config)[0]["command"]

    assert "--method scMAE+Harmony" in command
    assert "--n-pcs 20" in command
    assert "--harmony-max-iter 3" in command
    assert "--param sc_mae:epochs=2" in command


def test_manual_inductive_protocol_builds_group_command(tmp_path):
    config = ManualProtocolConfig(
        protocol="inductive",
        data=str(tmp_path / "toy.h5ad"),
        output_root=str(tmp_path / "out"),
        dataset_key="toy",
        label_key="Group",
        batch_key="donor",
        seeds=(7,),
        device="cpu",
        python_bin="python3",
        inductive_algorithms=("scraw", "pca_harmony"),
        split_key="donor",
        train_batches=("d1", "d2"),
        test_batches=("d3",),
        params=("scraw:training.epochs=2", "pca_harmony:n_pcs=5"),
    )

    command = build_jobs(config)[0]["command"]

    assert "run_shared_train_inductive_algorithms.py" in command
    assert "--train-batches d1 d2" in command
    assert "--test-batches d3" in command
    assert "--param scraw:training.epochs=2" in command
