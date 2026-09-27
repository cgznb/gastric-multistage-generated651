"""Independent result/provenance audit and input-only subprocess inference checks."""

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.umask(0o077)
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score  # noqa: E402

from stageworld.artifacts import atomic_write_private_json, read_json  # noqa: E402
from stageworld.regularization_inference import predict_bundle  # noqa: E402
from stageworld.regularization_spec import arms, specification  # noqa: E402
from stageworld.surgery_s2_data import load_verified_pool, subset_pool  # noqa: E402
from stageworld.surgery_s2_io import object_hash, sha256, source_hash  # noqa: E402
from stageworld.surgery_s2_spec import SOURCE_ROOT  # noqa: E402
from stageworld.surgery_s2_splits import validate_split  # noqa: E402
from stageworld.synthetic_workflow import _atomic_torch_save  # noqa: E402


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def independent_metrics(p, y):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel())
    n = len(y)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n,
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else None,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "bce": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "npv": tn / (tn + fn) if tn + fn else None,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
    }


def isolated(args):
    allowed = {Path(p).resolve() for p in (args.bundle, args.query, args.reference, args.output)}
    output = Path(args.output).resolve()
    library, code = Path(sys.prefix).resolve(), PROJECT / "src"

    def guard(event, items):
        if event != "open" or isinstance(items[0], int):
            return
        path = Path(os.fsdecode(items[0])).resolve()
        result_write = (
            path.parent == output.parent
            and path.name.startswith("." + output.name + ".")
            and items[2] & (os.O_WRONLY | os.O_RDWR)
        )
        if path.is_relative_to(Path.home()) and not (
            path in allowed
            or path.is_relative_to(library)
            or path.is_relative_to(code)
            or result_write
        ):
            raise PermissionError("Training/target access prohibited during standalone inference")

    sys.addaudithook(guard)
    denied = False
    try:
        load(SOURCE_ROOT / "artifacts/pool/pool.pt")
    except PermissionError:
        denied = True
    assert denied
    prediction = predict_bundle(Path(args.bundle), **load(Path(args.query)))
    reference = load(Path(args.reference))["logits"].double()
    torch.testing.assert_close(prediction["logits"], reference, atol=3e-5, rtol=1e-5)
    atomic_write_private_json(
        output,
        {
            "status": "passed",
            "training_data_access_blocked": True,
            "target_required": False,
            "patients": len(reference),
            "max_logit_error": float((prediction["logits"] - reference).abs().max()),
        },
    )


def audit(mode, arm):
    os.environ["GENERATED651_ARM"] = arm
    root = PROJECT / "artifacts" / f"{arm}-{mode}"
    spec = read_json(root / "protocol.json")
    assert spec["arms"] == specification(mode == "smoke")["arms"]
    assert spec["source_sha256"] == source_hash(PROJECT)
    imports = read_json(root / "imports.json")
    assert spec["imports_id"] == object_hash(imports)
    for entry in imports["entries"]:
        assert sha256(root / entry["path"]) == entry["sha256"]
    if mode == "formal":
        assert read_json(root / "completed.json")["status"] == "completed"
        lock = read_json(root / "selection_lock.json")
        assert lock["fit_count"] == len(spec["seeds"]) * len(spec["arms"])
        assert lock["protocol_id"] == object_hash(spec)
        for entry in lock["entries"]:
            assert sha256(root / entry["path"]) == entry["sha256"]
    else:
        assert not (root / "predictions").exists()
    pool = load_verified_pool()
    scores = {config["name"]: [] for config in spec["arms"]}
    checked, surgery_gradients, parity = 0, [], []
    for seed in spec["seeds"]:
        split = read_json(root / f"partitions/seed-{seed}/split.json")
        validate_split(pool, split)
        groups = split["patient_ids"]
        snapshot, anchor = (
            load(root / f"{part}/seed-{seed}.pt") for part in ("inputs", "anchors")
        )
        assert snapshot["fit_ids"] == groups["train"]
        for config in spec["arms"]:
            phase = root / "fits" / config["name"] / f"seed-{seed}"
            world_root = root / "world" / f"bs{config['batch_size']}" / f"seed-{seed}"
            assert read_json(world_root / "completed.json")["status"] == "completed"
            parent = load(world_root / "selected.pt")
            assert parent["contract"]["architecture"] == "world"
            assert parent["contract"]["batch_size"] == config["batch_size"]
            assert parent["contract"]["protocol_id"] == object_hash(spec)
            selected, final = load(phase / "selected.pt"), load(phase / "final.pt")
            report = read_json(phase / "completed.json")
            contract, state = selected["contract"], selected["model_state"]
            assert contract["protocol_id"] == object_hash(spec) and contract["arm"] == config
            assert contract["seed"] == seed and contract["parent_id"] == parent["artifact_id"]
            assert (
                contract["train_ids"] == groups["train"]
                and contract["validation_ids"] == groups["validation"]
            )
            assert (
                contract["snapshot_id"] == snapshot["artifact_id"]
                and contract["anchor_id"] == anchor["artifact_id"]
            )
            assert torch.equal(state["anchor_weight"], anchor["weight"])
            assert torch.equal(state["anchor_bias"], anchor["bias"])
            assert torch.equal(state["residual_scale"], torch.full((2,), config["alpha"]))
            block_indices = {int(k.split(".")[2]) for k in state if k.startswith("surgery.blocks.")}
            assert block_indices == set(range(config["surgery_blocks"]))
            assert not torch.equal(
                state["world.state_delta.1.weight"], parent["model_state"]["state_delta.1.weight"]
            )
            assert state["surgery.delta.1.weight"].any()
            assert report["surgery_gradient_norm_max"] > 0
            surgery_gradients.append(report["surgery_gradient_norm_max"])
            assert "optimizer_state" not in final and not (phase / "recovery.pt").exists()
            history = final["history"]
            assert history == read_json(phase / "history.json")["epochs"]
            assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
            best = min(history, key=lambda h: h["score"])
            assert best["epoch"] == selected["epoch"] == report["selected_epoch"]
            assert report["selected_id"] == selected["artifact_id"]
            validation = load(phase / "validation.pt")
            assert validation["selected_id"] == selected["artifact_id"]
            assert validation["patient_ids"] == groups["validation"]
            assert torch.equal(
                validation["labels"], pool.base.labels[pool.base.indices(groups["validation"])]
            )
            p, y = validation["logits"].double().sigmoid().numpy(), validation["labels"].numpy()
            ap = np.mean([average_precision_score(y[:, i], p[:, i]) for i in range(2)])
            assert math.isclose(-ap, best["score"], abs_tol=1e-12)
            assert math.isclose(ap, report["validation"]["mean_AP"], abs_tol=1e-12)
            scores[config["name"]].append(float(ap))
            for h in history:
                assert h["learning_rates"] == [
                    config["world_lr"],
                    config["head_surgery_lr"],
                    config["head_surgery_lr"],
                ]
                assert h["batches"] == -(-521 // config["batch_size"])
                assert 0 <= h["clip_fraction"] <= 1
                assert h["clip_fraction"] == h["clipped_batches"] / h["batches"]
                online = h["online"]
                assert math.isclose(
                    online["joint_loss"],
                    online["pcr_balanced_bce"]
                    + online["recurrence_balanced_bce"]
                    + config["ct_weight"] * online["ct_loss"],
                    abs_tol=2e-6,
                )
            checked += 1
    metric_count = 0
    if mode == "formal":
        choice = read_json(root / "configuration_choice.json")
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        assert choice["selected_arm"] == max(means, key=means.get) and choice["test_used"] is False
        rows = read_csv(root / "evaluation/per_seed_test_metrics.csv")
        expected_rows = len(spec["seeds"]) * (len(spec["arms"]) + 1) * 2
        assert len(rows) == expected_rows
        assert len({(r["seed"], r["arm"], r["endpoint"]) for r in rows}) == expected_rows
        lookup = {}
        for row in rows:
            seed, arm, endpoint = int(row["seed"]), row["arm"], row["endpoint"]
            saved = load(root / "predictions" / arm / f"seed-{seed}.pt")
            groups = read_json(root / f"partitions/seed-{seed}/split.json")["patient_ids"]
            assert saved["patient_ids"] == groups["test"] and saved["partition"] == "test"
            assert torch.equal(saved["labels"], pool.base.labels[pool.base.indices(groups["test"])])
            assert torch.equal(saved["probabilities"], saved["logits"].double().sigmoid())
            i = ("pcr", "recurrence").index(endpoint)
            actual = independent_metrics(
                saved["probabilities"][:, i].numpy(), saved["labels"][:, i].numpy()
            )
            for key, value in actual.items():
                assert (
                    row[key] == ""
                    if value is None
                    else math.isclose(float(row[key]), value, abs_tol=1e-12)
                )
            lookup[seed, arm, endpoint] = actual
            metric_count += 1
        for row in read_csv(root / "evaluation/seed_summary.csv"):
            values = [lookup[s, row["arm"], row["endpoint"]][row["metric"]] for s in spec["seeds"]]
            values = [v for v in values if v is not None]
            assert int(row["defined_seeds"]) == len(values)
            if values:
                assert math.isclose(float(row["mean"]), float(np.mean(values)), abs_tol=1e-12)
            if len(values) > 1:
                assert math.isclose(float(row["sd"]), float(np.std(values, ddof=1)), abs_tol=1e-12)
        paired = read_csv(root / "evaluation/paired_vs_baseline.csv")
        assert len(paired) == len(spec["seeds"]) * (len(spec["arms"]) - 1) * 2
        for row in paired:
            a = lookup[int(row["seed"]), row["arm"], row["endpoint"]]
            b = lookup[int(row["seed"]), spec["paired_baseline"], row["endpoint"]]
            for key, value in row.items():
                if key.startswith("delta_"):
                    assert math.isclose(float(value), a[key[6:]] - b[key[6:]], abs_tol=1e-12)
    isolated_reports = []
    for name in ("bs16_baseline", "bs16_lr1_alpha05", "bs16_alpha05_s2b2", "bs16_lr1_alpha05_s2b2"):
        seed = spec["seeds"][0]
        phase = root / "fits" / name / f"seed-{seed}"
        validation = load(phase / "validation.pt")
        sample = subset_pool(pool, validation["patient_ids"])
        folder = root / "verification/isolated" / name
        query, reference, output = (
            folder / "query.pt",
            folder / "reference.pt",
            folder / "result.json",
        )
        _atomic_torch_save(
            query,
            {
                "clinical": [sample.base.clinical[p] for p in sample.base.ids],
                "treatments": [sample.base.treatments[p] for p in sample.base.ids],
                "interval": sample.base.interval,
                "ct0": sample.base.ct0,
                "surgery": sample.surgery,
            },
        )
        _atomic_torch_save(reference, {"logits": validation["logits"]})
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "isolated",
                "--bundle",
                str(phase / "inference.pt"),
                "--query",
                str(query),
                "--reference",
                str(reference),
                "--output",
                str(output),
            ],
            check=True,
            cwd=PROJECT,
        )
        isolated_reports.append({"arm": name, **read_json(output)})
    result = {
        "status": "passed",
        "fits_verified": checked,
        "validation_scores_recomputed": checked,
        "test_endpoint_rows_recomputed": metric_count,
        "minimum_surgery_gradient": min(surgery_gradients),
        "isolated_inference": isolated_reports,
        "source_binding_verified": True,
        "imports_verified": len(imports["entries"]),
    }
    atomic_write_private_json(root / "verification/independent_audit.json", result)
    print(
        {k: result[k] for k in ("status", "fits_verified", "test_endpoint_rows_recomputed")},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    audit_parser.add_argument("--arm", choices=["all", *[a["name"] for a in arms()]],
                              default="bs4_baseline")
    isolated_parser = commands.add_parser("isolated")
    for key in ("bundle", "query", "reference", "output"):
        isolated_parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "audit":
        audit(args.mode, args.arm)
    else:
        isolated(args)


if __name__ == "__main__":
    main()
