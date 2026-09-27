"""From-scratch two-stage training, paired workers, validation-only choice, gated testing."""

import gc
import subprocess
import sys
import time

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary700_statistics import logistic_anchor
from stageworld.generated651_evaluation import write_csv
from stageworld.generated651_workflow import fit_anchor
from stageworld.generated700_data import fit_inputs
from stageworld.regularization_inference import export_bundle, predict_bundle
from stageworld.regularization_metrics import metrics_from_predictions, predict
from stageworld.regularization_models import build_model
from stageworld.regularization_spec import specification
from stageworld.regularization_training import train_fit, train_world
from stageworld.surgery_s2_data import encode_inputs, load_verified_pool, subset_pool
from stageworld.surgery_s2_evaluation import endpoint_metrics
from stageworld.surgery_s2_io import bound_json, object_hash, sha256, source_hash
from stageworld.surgery_s2_spec import EVENT_ROOT, SOURCE_ROOT
from stageworld.surgery_s2_splits import make_split, validate_split
from stageworld.synthetic_workflow import _atomic_torch_save


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def prepare(project, root, smoke):
    for path, key in (
        (SOURCE_ROOT / "artifacts/pool/pool.pt", "pool_sha256"),
        (EVENT_ROOT / "artifacts/pool/events.pt", "events_sha256"),
    ):
        if not path.exists():
            raise ValueError("Read-only feature or event cache missing")
    pool = load_verified_pool()
    spec = specification(smoke)
    spec.update(
        {
            "source_sha256": source_hash(project),
            "pool_sha256": sha256(SOURCE_ROOT / "artifacts/pool/pool.pt"),
            "events_sha256": sha256(EVENT_ROOT / "artifacts/pool/events.pt"),
            "pool_id": pool.base.artifact_id,
            "event_pool_id": pool.artifact_id,
            "torch_version": str(torch.__version__),
        }
    )
    manifest = {"schema": "fullretrain-imports-v1", "entries": []}
    bound_json(root / "imports.json", manifest)
    spec["imports_id"] = object_hash(manifest)
    bound_json(root / "protocol.json", spec)
    counts = []
    for seed in spec["seeds"]:
        split = make_split(pool, seed)
        bound_json(root / "partitions" / f"seed-{seed}" / "split.json", split)
        validate_split(pool, split)
        counts.append({"seed": seed, "counts": split["counts"]})
    bound_json(
        root / "data_audit.json",
        {
            "patients": len(pool.base.ids),
            "surgery_present": int((pool.surgery == 1).sum()),
            "pcr_positive": int(pool.base.labels[:, 0].sum()),
            "recurrence_positive": int(pool.base.labels[:, 1].sum()),
            "ct0_shape": list(pool.base.ct0.shape),
            "ct1_shape": list(pool.base.ct1_tokens.shape),
            "event_source_binding_verified": True,
            "patient_identity_join_verified": True,
            "counts_by_seed": counts,
            "test_policy": spec["test_gate"],
        },
    )
    return pool, spec


def seed_context(pool, root, seed):
    split = read_json(root / f"partitions/seed-{seed}/split.json")
    validate_split(pool, split)
    groups = split["patient_ids"]
    dev = subset_pool(pool, groups["train"] + groups["validation"])
    if set(dev.base.ids) & set(groups["test"]):
        raise ValueError("Test patients entered development")
    x, snapshot = fit_inputs(dev.base, groups["train"], root / "inputs" / f"seed-{seed}.pt")
    train = dev.base.indices(groups["train"])
    validation = dev.base.indices(groups["validation"])
    path = root / "anchors" / f"seed-{seed}.pt"
    contract = {
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": groups["train"],
        "C": 1.0,
        "class_weight": None,
        "surgery_input": False,
    }
    if path.exists():
        anchor = torch.load(path, weights_only=True, map_location="cpu")
        if anchor["contract"] != contract:
            raise ValueError("Logistic anchor partition differs")
    else:
        weight, bias = logistic_anchor(fit_anchor(dev.base, x, train))
        anchor = {
            "artifact_id": new_artifact_id("fullretrain-anchor"),
            "contract": contract,
            "weight": weight,
            "bias": bias,
        }
        _atomic_torch_save(path, anchor)
    return dev, x, snapshot, train, validation, anchor


def phase_dir(root, seed, arm):
    return root / "fits" / arm / f"seed-{seed}"


def world_dir(root, seed, batch_size):
    return root / "world" / f"bs{batch_size}" / f"seed-{seed}"


def train_all(pool, root, spec, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(spec["seeds"]):
        raise ValueError("Worker seed assignment differs")
    for seed in seeds:
        dev, x, snapshot, train, validation, anchor = seed_context(pool, root, seed)
        by_batch = {}
        for config in spec["arms"]:
            by_batch.setdefault(config["batch_size"], []).append(config)
        for batch_size in spec["batch_sizes"]:
            world_root = world_dir(root, seed, batch_size)
            world, world_report = train_world(
                dev,
                x,
                snapshot,
                train,
                validation,
                world_root,
                seed=seed,
                batch_size=batch_size,
                spec=spec,
            )
            del world
            for config in by_batch[batch_size]:
                phase = phase_dir(root, seed, config["name"])
                model, report = train_fit(
                    dev,
                    x,
                    snapshot,
                    train,
                    validation,
                    phase,
                    seed=seed,
                    config=config,
                    spec=spec,
                    anchor=anchor,
                    parent=load(world_root / "selected.pt"),
                )
                if not (phase / "inference.pt").exists() or not (phase / "validation.pt").exists():
                    output = predict(model, x, dev, validation)
                    _atomic_torch_save(
                        phase / "validation.pt",
                        {
                            "partition": "validation",
                            "selected_id": report["selected_id"],
                            "patient_ids": [dev.base.ids[i] for i in validation.tolist()],
                            "logits": output["logits"],
                            "labels": dev.base.labels[validation],
                        },
                    )
                    export_bundle(
                        phase / "inference.pt", snapshot, model, config, report, object_hash(spec)
                    )
                del model
                gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del dev, x


def freeze_selection(root, spec):
    entries, values = [], {config["name"]: [] for config in spec["arms"]}
    for seed in spec["seeds"]:
        for batch_size in spec["batch_sizes"]:
            world_root = world_dir(root, seed, batch_size)
            if not (world_root / "completed.json").exists() or not (world_root / "selected.pt").exists():
                raise ValueError("All world pretrains must complete before test scoring")
            entries.append(
                {
                    "path": str((world_root / "selected.pt").relative_to(root)),
                    "sha256": sha256(world_root / "selected.pt"),
                }
            )
        for config in spec["arms"]:
            phase = phase_dir(root, seed, config["name"])
            if not (phase / "completed.json").exists():
                raise ValueError("All declared fits must complete before test scoring")
            report, selected = read_json(phase / "completed.json"), load(phase / "selected.pt")
            if (
                report["status"] != "completed"
                or report["selected_id"] != selected["artifact_id"]
                or selected["contract"]["protocol_id"] != object_hash(spec)
                or selected["contract"]["arm"] != config
                or selected["contract"]["seed"] != seed
            ):
                raise ValueError("Selected checkpoint registry differs")
            values[config["name"]].append(report["validation"]["mean_AP"])
            for filename in (
                "selected.pt",
                "inference.pt",
                "validation.pt",
                "completed.json",
                "history.json",
            ):
                path = phase / filename
                if not path.exists():
                    raise ValueError("Selection requires complete validation and inference exports")
                entries.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    manifest = read_json(root / "imports.json")
    if object_hash(manifest) != spec["imports_id"]:
        raise ValueError("Imported artifact manifest differs")
    for entry in manifest["entries"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError("A frozen parent, split or preprocessing artifact changed")
    if source_hash(root.parents[1]) != spec["source_sha256"]:
        raise ValueError("Training source changed before selection was locked")
    scores = [
        {
            "arm": config["name"],
            "mean_validation_AP": float(np.mean(values[config["name"]])),
            "sd_validation_AP": float(np.std(values[config["name"]], ddof=1))
            if len(spec["seeds"]) > 1
            else 0.0,
            "seeds": len(values[config["name"]]),
        }
        for config in spec["arms"]
    ]
    winner = max(scores, key=lambda r: r["mean_validation_AP"])["arm"]
    choice = {
        "selected_arm": winner,
        "criterion": spec["configuration_selection"],
        "tie_break": spec["configuration_tie_break"],
        "scores": scores,
        "test_used": False,
    }
    result = {
        "schema": "fullretrain-selection-lock-v1",
        "protocol_id": object_hash(spec),
        "fit_count": len(spec["seeds"]) * len(spec["arms"]),
        "choice": choice,
        "entries": entries,
    }
    bound_json(root / "selection_lock.json", result)
    bound_json(root / "configuration_choice.json", choice)
    return result


def replay_bundle(phase, sample):
    return predict_bundle(
        phase / "inference.pt",
        [sample.base.clinical[p] for p in sample.base.ids],
        [sample.base.treatments[p] for p in sample.base.ids],
        sample.base.interval,
        sample.base.ct0,
        sample.surgery,
    )


def write_results(root, spec, rows, diagnostics, replay_errors):
    expected = len(spec["seeds"]) * (len(spec["arms"]) + 1) * 2
    if (
        len(rows) != expected
        or len({(r["seed"], r["arm"], r["endpoint"]) for r in rows}) != expected
    ):
        raise ValueError("Test result coverage differs")
    output = root / "evaluation"
    output.mkdir(exist_ok=True, mode=0o700)
    write_csv(output / "per_seed_test_metrics.csv", rows)
    write_csv(output / "test_diagnostics.csv", diagnostics)
    lookup = {(r["seed"], r["arm"], r["endpoint"]): r for r in rows}
    summary, paired, wide = [], [], []
    all_arms = [config["name"] for config in spec["arms"]] + ["logistic"]
    base_arm = spec["paired_baseline"]
    for arm in all_arms:
        for endpoint in ("pcr", "recurrence"):
            for metric in (
                "auroc",
                "auprc",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "mcc",
                "bce",
                "brier",
            ):
                numbers = [lookup[s, arm, endpoint][metric] for s in spec["seeds"]]
                numbers = [n for n in numbers if n is not None]
                summary.append(
                    {
                        "arm": arm,
                        "endpoint": endpoint,
                        "metric": metric,
                        "mean": float(np.mean(numbers)) if numbers else None,
                        "sd": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
                        "defined_seeds": len(numbers),
                    }
                )
        for seed in spec["seeds"]:
            row = {"arm": arm, "seed": seed, "train_n": 521, "validation_n": 65, "test_n": 65}
            for endpoint in ("pcr", "recurrence"):
                item = lookup[seed, arm, endpoint]
                row.update(
                    {
                        f"{endpoint}_{k}": v
                        for k, v in item.items()
                        if k not in ("seed", "arm", "endpoint", "partition")
                    }
                )
                if arm not in (base_arm, "logistic"):
                    base = lookup[seed, base_arm, endpoint]
                    paired.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "endpoint": endpoint,
                            **{
                                f"delta_{k}": item[k] - base[k]
                                for k in (
                                    "auroc",
                                    "auprc",
                                    "brier",
                                    "bce",
                                    "sensitivity",
                                    "specificity",
                                )
                            },
                        }
                    )
            wide.append(row)
    write_csv(output / "seed_summary.csv", summary)
    write_csv(output / "paired_vs_baseline.csv", paired)
    write_csv(output / "all_configs_per_seed.csv", wide)
    choice = read_json(root / "configuration_choice.json")
    result = {
        "status": "completed",
        "seeds": len(spec["seeds"]),
        "configurations": len(spec["arms"]),
        "test_endpoint_rows": len(rows),
        "choice": choice,
        "summary": summary,
    }
    atomic_write_private_json(output / "summary.json", result)
    atomic_write_private_json(
        root / "verification/bundle_replay.json",
        {"status": "passed", "count": len(replay_errors), "results": replay_errors},
    )
    return result


@torch.inference_mode()
def evaluate_tests(pool, root, spec):
    if spec["smoke"]:
        raise ValueError("Smoke cannot score test patients")
    lock = freeze_selection(root, spec)
    for entry in lock["entries"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError("Locked selected artifact changed")
    rows, diagnostics, errors = [], [], []
    atomic_write_private_json(
        root / "progress.json", {"status": "evaluating", "test_scoring_started": True}
    )
    for seed in spec["seeds"]:
        groups = read_json(root / f"partitions/seed-{seed}/split.json")["patient_ids"]
        test = subset_pool(pool, groups["test"])
        snapshot, anchor = (
            load(root / f"inputs/seed-{seed}.pt"),
            load(root / f"anchors/seed-{seed}.pt"),
        )
        x, indices = encode_inputs(test.base, snapshot), torch.arange(len(test.base.ids))
        z = torch.nn.functional.linear(x.float(), anchor["weight"], anchor["bias"])
        for arm in [None, *spec["arms"]]:
            name = "logistic" if arm is None else arm["name"]
            selected_id = anchor["artifact_id"]
            if arm is not None:
                phase = phase_dir(root, seed, name)
                selected = load(phase / "selected.pt")
                model = build_model(arm, test.base.ct0.shape[-1], seed, anchor).to("cuda")
                model.load_state_dict(selected["model_state"], strict=True)
                output = predict(model, x, test, indices)
                z, selected_id = output["logits"], selected["artifact_id"]
                metrics, _ = metrics_from_predictions(
                    output, test, indices, selected["positive_weights"], arm["ct_weight"]
                )
                diagnostics.append({"seed": seed, "arm": name, **metrics})
                standalone = replay_bundle(phase, test)
                error = float((standalone["logits"] - z.double()).abs().max())
                if not torch.allclose(standalone["logits"], z.double(), atol=3e-5, rtol=1e-5):
                    raise ValueError("Standalone test inference differs")
                errors.append({"seed": seed, "arm": name, "max_logit_difference": error})
                del model
            probability = z.double().sigmoid()
            _atomic_torch_save(
                root / "predictions" / name / f"seed-{seed}.pt",
                {
                    "partition": "test",
                    "seed": seed,
                    "arm": name,
                    "protocol_id": object_hash(spec),
                    "selected_id": selected_id,
                    "patient_ids": test.base.ids,
                    "logits": z,
                    "probabilities": probability,
                    "labels": test.base.labels,
                },
            )
            rows.extend(
                {"seed": seed, "arm": name, "endpoint": endpoint, "partition": "test", **metrics}
                for endpoint, metrics in endpoint_metrics(probability, test.base.labels).items()
            )
        print(f"Test evaluation and standalone replay completed: seed={seed}", flush=True)
    return write_results(root, spec, rows, diagnostics, errors)


def run_study(project, root, *, smoke, action, workers=2, worker_seeds=None):
    began = time.monotonic()
    pool, spec = prepare(project, root, smoke)
    if action == "prepare":
        return {
            "status": "prepared",
            "configurations": len(spec["arms"]),
            "seeds": len(spec["seeds"]),
        }
    if action == "evaluate":
        return evaluate_tests(pool, root, spec)
    if worker_seeds is not None:
        train_all(pool, root, spec, worker_seeds)
        return {"status": "worker_completed"}
    if workers > 1 and len(spec["seeds"]) > 1:
        processes = []
        try:
            for i in range(workers):
                assignment = spec["seeds"][i::workers]
                command = [
                    sys.executable,
                    "-u",
                    str(project / "scripts/run_regularization.py"),
                    "--mode",
                    "smoke" if smoke else "formal",
                    "--worker-seeds",
                    ",".join(map(str, assignment)),
                ]
                processes.append(subprocess.Popen(command, cwd=project))
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("A regularization worker failed; recovery retained")
                atomic_write_private_json(
                    root / "progress.json",
                    {
                        "status": "training",
                        "completed_fits": len(list(root.glob("fits/*/seed-*/completed.json"))),
                        "total_fits": len(spec["seeds"]) * len(spec["arms"]),
                        "test_scoring_started": False,
                        "seconds": time.monotonic() - began,
                    },
                )
                time.sleep(5)
            if any(p.returncode for p in processes):
                raise RuntimeError("A regularization worker exited unsuccessfully")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
    else:
        train_all(pool, root, spec, spec["seeds"])
    if smoke:
        seed = spec["seeds"][0]
        dev, _, _, _, validation, _ = seed_context(pool, root, seed)
        sample = subset_pool(dev, [dev.base.ids[i] for i in validation.tolist()])
        for arm in spec["arms"]:
            phase = phase_dir(root, seed, arm["name"])
            expected = load(phase / "validation.pt")["logits"].double()
            actual = replay_bundle(phase, sample)["logits"]
            if not torch.allclose(expected, actual, atol=3e-5, rtol=1e-5):
                raise ValueError("Smoke validation bundle mismatch")
        result = {"status": "smoke_completed", "fits": len(spec["arms"]), "test_evaluated": False}
    else:
        result = evaluate_tests(pool, root, spec)
    result["seconds"] = time.monotonic() - began
    atomic_write_private_json(root / "completed.json", result)
    atomic_write_private_json(root / "progress.json", {k: result[k] for k in ("status", "seconds")})
    return result
