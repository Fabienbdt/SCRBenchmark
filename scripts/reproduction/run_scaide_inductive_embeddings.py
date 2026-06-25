#!/usr/bin/env python3
"""Train AIDE on a train matrix and project held-out matrices in one TF1 session."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AIDE_ROOT = Path(
    os.environ.get(
        "SCAIDE_ROOT",
        REPO_ROOT / "external" / "original_code" / "aide",
    )
)


def patch_runtime() -> None:
    import joblib
    import sklearn.externals
    import tensorflow.compat.v1 as tf

    sklearn.externals.joblib = joblib
    tf.disable_v2_behavior()

    contrib_layers = types.SimpleNamespace(
        xavier_initializer=lambda uniform=True: tf.glorot_uniform_initializer(),
        l2_regularizer=lambda scale: (lambda w: tf.multiply(float(scale), tf.nn.l2_loss(w))),
        apply_regularization=lambda regularizer, weights_list=None: (
            tf.add_n([regularizer(w) for w in (weights_list or tf.trainable_variables())])
            if (weights_list or tf.trainable_variables())
            else tf.constant(0.0)
        ),
    )
    tf.contrib = types.SimpleNamespace(layers=contrib_layers)
    sys.modules["tensorflow"] = tf

    if str(AIDE_ROOT) not in sys.path:
        sys.path.insert(0, str(AIDE_ROOT))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("NAME cannot be empty.")
    return name, Path(raw_path).expanduser().resolve()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_payload(config: Any) -> dict[str, Any]:
    return {str(key): safe_json(value) for key, value in vars(config).items()}


def tune_config(config: Any, n_obs: int, n_vars: int, fast_smoke: bool) -> None:
    if n_obs < int(config.validate_size):
        config.validate_size = max(1, min(int(n_obs // 2), int(config.batch_size)))
    if n_obs < int(config.embed_batch_size):
        config.embed_batch_size = max(1, int(n_obs))

    if fast_smoke:
        config.pretrain_step_num = 1
        config.min_step_num = 1
        config.max_step_num = 2
        config.val_freq = 1
        config.print_freq = 1
        config.batch_size = min(16, max(2, int(n_obs // 4)))
        config.validate_size = min(max(1, int(n_obs // 2)), 64)
        config.embed_batch_size = min(max(1, int(n_obs)), 256)
        config.ae_units = [min(128, max(8, n_vars * 2)), min(64, max(4, n_vars))]
        config.ae_acts = ["relu", None]
        config.mds_units = [min(128, max(8, n_vars * 2)), min(64, max(4, n_vars))]
        config.mds_acts = ["relu", None]
        config.verbose = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input-npy", required=True)
    parser.add_argument("--train-output-npy", required=True)
    parser.add_argument("--test-input", action="append", default=[], type=parse_named_path)
    parser.add_argument("--test-output", action="append", default=[], type=parse_named_path)
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--name", default="scaide_inductive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-smoke", action="store_true")
    args = parser.parse_args()

    test_inputs = dict(args.test_input)
    test_outputs = dict(args.test_output)
    if set(test_inputs) != set(test_outputs):
        missing_outputs = sorted(set(test_inputs) - set(test_outputs))
        missing_inputs = sorted(set(test_outputs) - set(test_inputs))
        raise SystemExit(
            "Mismatched test inputs/outputs; "
            f"missing_outputs={missing_outputs} missing_inputs={missing_inputs}"
        )

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    np.random.seed(int(args.seed))
    patch_runtime()

    import tensorflow.compat.v1 as tf
    from aide import AIDE, AIDEConfig
    from aide.utils_ import x_to_input

    tf.set_random_seed(int(args.seed))

    train_input = Path(args.train_input_npy).expanduser().resolve()
    train_output = Path(args.train_output_npy).expanduser().resolve()
    save_folder = Path(args.save_folder).expanduser().resolve()
    train_output.parent.mkdir(parents=True, exist_ok=True)
    save_folder.mkdir(parents=True, exist_ok=True)

    x_train = np.load(train_input).astype(np.float32, copy=False)
    config = AIDEConfig()
    tune_config(config, int(x_train.shape[0]), int(x_train.shape[1]), bool(args.fast_smoke))

    encoder = AIDE(name=str(args.name), save_folder=str(save_folder))
    train_embedding = encoder.fit_transform(x_train, config=config)
    np.save(train_output, np.asarray(train_embedding, dtype=np.float32))

    test_shapes: dict[str, list[int]] = {}
    for name, input_path in sorted(test_inputs.items()):
        output_path = test_outputs[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        x_test = np.load(input_path).astype(np.float32, copy=False)
        if x_test.shape[1] != x_train.shape[1]:
            raise ValueError(
                f"Test matrix {name!r} has {x_test.shape[1]} features; expected {x_train.shape[1]}."
            )
        encoder.pred_feed = x_to_input(x_test)
        test_embedding = encoder.get_embedding()
        np.save(output_path, np.asarray(test_embedding, dtype=np.float32))
        test_shapes[name] = list(x_test.shape)

    manifest = {
        "train_input_npy": str(train_input),
        "train_output_npy": str(train_output),
        "save_folder": str(save_folder),
        "name": str(args.name),
        "seed": int(args.seed),
        "fast_smoke": bool(args.fast_smoke),
        "train_shape": list(x_train.shape),
        "train_embedding_shape": list(np.asarray(train_embedding).shape),
        "test_shapes": test_shapes,
        "test_outputs": {name: str(path) for name, path in sorted(test_outputs.items())},
        "aide_config": config_payload(config),
    }
    (train_output.parent / "scaide_inductive_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
