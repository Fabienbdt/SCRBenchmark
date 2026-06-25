#!/usr/bin/env python3
"""Compute a scAIDE/AIDE embedding from a dense NumPy matrix.

The upstream AIDE code is TensorFlow-1 style. This wrapper provides the small
compatibility shims needed to run the local source tree with the TF2 runtime
available in ``envs/desc_py311``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np


RUNNER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("SCRBENCHMARK_ROOT", Path(__file__).resolve().parents[3])).resolve()
AIDE_ROOT = Path(
    os.environ.get(
        "AIDE_ROOT",
        REPO_ROOT / "external" / "original_code" / "aide",
    )
).resolve()


def _patch_runtime() -> None:
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


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _config_payload(config: Any) -> Dict[str, Any]:
    return {str(k): _safe_json(v) for k, v in vars(config).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--save-folder", required=True)
    parser.add_argument("--name", default="scaide")
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Use a tiny AIDE config for import/runtime smoke tests only.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _patch_runtime()

    from aide import AIDE, AIDEConfig

    input_path = Path(args.input_npy).resolve()
    output_path = Path(args.output_npy).resolve()
    save_folder = Path(args.save_folder).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_folder.mkdir(parents=True, exist_ok=True)

    X = np.load(input_path).astype(np.float32, copy=False)
    config = AIDEConfig()

    # Avoid impossible validation batches on small datasets while preserving
    # upstream defaults for normal-sized inputs.
    if X.shape[0] < int(config.validate_size):
        config.validate_size = max(1, min(int(X.shape[0] // 2), int(config.batch_size)))
    if X.shape[0] < int(config.embed_batch_size):
        config.embed_batch_size = max(1, int(X.shape[0]))

    if args.fast_smoke:
        config.pretrain_step_num = 1
        config.min_step_num = 1
        config.max_step_num = 2
        config.val_freq = 1
        config.print_freq = 1
        config.batch_size = min(16, max(2, int(X.shape[0] // 4)))
        config.validate_size = min(max(1, int(X.shape[0] // 2)), 64)
        config.embed_batch_size = min(max(1, int(X.shape[0])), 256)
        config.ae_units = [min(128, max(8, X.shape[1] * 2)), min(64, max(4, X.shape[1]))]
        config.ae_acts = ["relu", None]
        config.mds_units = [min(128, max(8, X.shape[1] * 2)), min(64, max(4, X.shape[1]))]
        config.mds_acts = ["relu", None]
        config.verbose = False

    encoder = AIDE(name=str(args.name), save_folder=str(save_folder))
    embedding = encoder.fit_transform(X, config=config)
    np.save(output_path, np.asarray(embedding, dtype=np.float32))

    manifest = {
        "input_npy": str(input_path),
        "output_npy": str(output_path),
        "save_folder": str(save_folder),
        "name": str(args.name),
        "input_shape": list(X.shape),
        "embedding_shape": list(np.asarray(embedding).shape),
        "fast_smoke": bool(args.fast_smoke),
        "aide_config": _config_payload(config),
    }
    (output_path.parent / "scaide_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
