"""Run every declared fit, freeze selection globally, then evaluate patient holdouts."""

import gc
import subprocess
import sys
import time
from pathlib import Path

import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary700_statistics import logistic_anchor
from stageworld.generated651_workflow import fit_anchor
from stageworld.generated700_data import fit_inputs
from stageworld.surgery_s2_data import encode_inputs, load_verified_pool, subset_pool
from stageworld.surgery_s2_evaluation import (
    ct_metrics,
    endpoint_metrics,
    state_diagnostics,
    write_reports,
)
from stageworld.surgery_s2_inference import export_bundle, predict_bundle
from stageworld.surgery_s2_io import bound_json, object_hash, sha256, source_hash
from stageworld.surgery_s2_spec import EVENT_ROOT, SOURCE_ROOT, specification
from stageworld.surgery_s2_splits import make_split, validate_split
from stageworld.surgery_s2_training import build_model, infer, train_phase
from stageworld.synthetic_workflow import _atomic_torch_save


def prepare(project: Path, root: Path, smoke: bool) -> tuple:
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


def seed_context(pool, root: Path, seed: int):
    split = read_json(root / "partitions" / f"seed-{seed}" / "split.json")
    validate_split(pool, split)
    groups = split["patient_ids"]
    dev = subset_pool(pool, groups["train"] + groups["validation"])
    if set(dev.base.ids) & set(groups["test"]):
        raise ValueError("Test patients entered the development object")
    transform = root / "inputs" / f"seed-{seed}.pt"
    x, snapshot = fit_inputs(dev.base, groups["train"], transform)
    train, validation = dev.base.indices(groups["train"]), dev.base.indices(groups["validation"])
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
            "artifact_id": new_artifact_id("surgery-s2-anchor"),
            "contract": contract,
            "weight": weight,
            "bias": bias,
        }
        _atomic_torch_save(path, anchor)
    return dev, x, snapshot, train, validation, anchor


def phase_directory(root: Path, seed: int, architecture: str, loss: str = "balanced") -> Path:
    if architecture == "world":
        return root / "world" / f"seed-{seed}"
    return root / "fits" / f"{architecture}_{loss}" / f"seed-{seed}"


def freeze_selection(root: Path, spec: dict) -> dict:
    entries = []
    for seed in spec["seeds"]:
        phases = [("world", "balanced")] + [
            (arch, loss) for loss in spec["losses"] for arch in spec["architectures"]
        ]
        for architecture, loss in phases:
            phase = phase_directory(root, seed, architecture, loss)
            if not (phase / "completed.json").exists():
                raise ValueError("Test scoring is blocked until every declared fit has completed")
            report = read_json(phase / "completed.json")
            selected = torch.load(phase / "selected.pt", weights_only=True, map_location="cpu")
            if (
                report["status"] != "completed"
                or report["selected_id"] != selected["artifact_id"]
                or selected["contract"]["protocol_id"] != object_hash(spec)
                or selected["contract"]["seed"] != seed
                or selected["contract"]["architecture"] != architecture
                or selected["contract"]["loss"] != loss
            ):
                raise ValueError("Selection registry binding differs")
            entries.append(
                {
                    "path": str((phase / "selected.pt").relative_to(root)),
                    "sha256": sha256(phase / "selected.pt"),
                    "selected_id": selected["artifact_id"],
                }
            )
            if architecture != "world":
                path = phase / "inference.pt"
                if not path.exists() or not (phase / "validation.pt").exists():
                    raise ValueError("Selection requires an exported bundle and validation replay")
                entries.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
        for path in (
            root / "inputs" / f"seed-{seed}.pt",
            root / "anchors" / f"seed-{seed}.pt",
            root / "partitions" / f"seed-{seed}" / "split.json",
        ):
            entries.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    lock = {
        "schema": "surgery-s2-selection-lock-v1",
        "protocol_id": object_hash(spec),
        "checkpoint_count": len(spec["seeds"]) * 7,
        "entries": entries,
    }
    bound_json(root / "selection_lock.json", lock)
    return lock


@torch.inference_mode()
def publish_validation(model, dev, x, snapshot, validation, phase, report, spec):
    logits = infer(model, x, dev, validation)
    probabilities = logits.double().sigmoid()
    _atomic_torch_save(
        phase / "validation.pt",
        {
            "partition": "validation",
            "selected_id": report["selected_id"],
            "patient_ids": [dev.base.ids[i] for i in validation.tolist()],
            "logits": logits,
            "probabilities": probabilities,
            "labels": dev.base.labels[validation],
        },
    )
    export_bundle(phase / "inference.pt", snapshot, model, report, object_hash(spec))
    if not (phase / "validation_metrics.json").exists():
        atomic_write_private_json(
            phase / "validation_metrics.json",
            {
                "endpoints": endpoint_metrics(probabilities, dev.base.labels[validation]),
                "states": state_diagnostics(model, x, dev, validation),
            },
        )


def train_all(pool, root: Path, spec: dict, seeds: list[int] | None = None) -> None:
    done = 0
    chosen = spec["seeds"] if seeds is None else seeds
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(spec["seeds"]):
        raise ValueError("Worker seed assignment differs from the registered seeds")
    expected = len(chosen) * 7
    for seed in chosen:
        dev, x, snapshot, train, validation, anchor = seed_context(pool, root, seed)
        world_root = phase_directory(root, seed, "world")
        print(f"Starting seed={seed}, train={len(train)}, validation={len(validation)}", flush=True)
        world, _ = train_phase(
            dev,
            x,
            snapshot,
            train,
            validation,
            world_root,
            seed=seed,
            architecture="world",
            loss_name="balanced",
            spec=spec,
        )
        del world
        done += 1
        for loss in spec["losses"]:
            for architecture in spec["architectures"]:
                phase = phase_directory(root, seed, architecture, loss)
                model, report = train_phase(
                    dev,
                    x,
                    snapshot,
                    train,
                    validation,
                    phase,
                    seed=seed,
                    architecture=architecture,
                    loss_name=loss,
                    spec=spec,
                    anchor=anchor,
                    parent_path=world_root / "selected.pt",
                )
                if not (phase / "inference.pt").exists() or not (phase / "validation.pt").exists():
                    publish_validation(model, dev, x, snapshot, validation, phase, report, spec)
                del model
                done += 1
                atomic_write_private_json(
                    root / "worker_progress" / f"seed-{seed}.json",
                    {
                        "status": "training",
                        "completed_phases": done,
                        "total_phases": expected,
                        "seed": seed,
                        "architecture": architecture,
                        "loss": loss,
                        "test_scoring_started": False,
                    },
                )
        del dev, x
        gc.collect()


@torch.inference_mode()
def evaluate_tests(pool, root: Path, spec: dict) -> dict:
    if spec["smoke"]:
        raise ValueError("Engineering smoke must never score the test partition")
    freeze_selection(root, spec)
    metrics_rows, ct_rows, diagnostic_rows = [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle_errors = []
    for seed in spec["seeds"]:
        split = read_json(root / "partitions" / f"seed-{seed}" / "split.json")
        validate_split(pool, split)
        test = subset_pool(pool, split["patient_ids"]["test"])
        snapshot = torch.load(
            root / "inputs" / f"seed-{seed}.pt", weights_only=True, map_location="cpu"
        )
        anchor = torch.load(
            root / "anchors" / f"seed-{seed}.pt", weights_only=True, map_location="cpu"
        )
        if snapshot["fit_ids"] != split["patient_ids"]["train"]:
            raise ValueError("Test preprocessing was not fitted on the bound train partition")
        x = encode_inputs(test.base, snapshot)
        test_rows = torch.arange(len(test.base.ids))
        train_rows = pool.base.indices(split["patient_ids"]["train"])
        mean_ct1 = pool.base.ct1_tokens[train_rows].mean(0)
        parent = torch.load(
            phase_directory(root, seed, "world") / "selected.pt",
            weights_only=True,
            map_location="cpu",
        )
        world = build_model("world", pool.base.ct0.shape[-1], seed).to(device)
        world.load_state_dict(parent["model_state"], strict=True)
        for comparator, values in ct_metrics(world, x, test, test_rows, mean_ct1).items():
            ct_rows.append(
                {
                    "seed": seed,
                    "arm": "pretrain",
                    "comparator": comparator,
                    "partition": "test",
                    **values,
                }
            )
        del world
        logits = torch.nn.functional.linear(x, anchor["weight"], anchor["bias"])
        predictions = [("logistic", logits.double().sigmoid(), anchor["artifact_id"], logits)]
        for loss in spec["losses"]:
            for architecture in spec["architectures"]:
                arm = f"{architecture}_{loss}"
                phase = phase_directory(root, seed, architecture, loss)
                selected = torch.load(phase / "selected.pt", weights_only=True, map_location="cpu")
                model = build_model(architecture, pool.base.ct0.shape[-1], seed, anchor, parent)
                model.load_state_dict(selected["model_state"], strict=True)
                model = model.to(device).eval()
                logits = infer(model, x, test, test_rows)
                standalone = predict_bundle(
                    phase / "inference.pt",
                    [test.base.clinical[p] for p in test.base.ids],
                    [test.base.treatments[p] for p in test.base.ids],
                    test.base.interval,
                    test.base.ct0,
                    test.surgery,
                )
                error = float((standalone["logits"] - logits.double()).abs().max())
                if not torch.allclose(standalone["logits"], logits.double(), atol=3e-5, rtol=1e-5):
                    raise ValueError(
                        "Standalone CPU inference differs from selected GPU checkpoint"
                    )
                bundle_errors.append({"seed": seed, "arm": arm, "max_logit_error": error})
                predictions.append(
                    (arm, logits.double().sigmoid(), selected["artifact_id"], logits)
                )
                diagnostic_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "partition": "test",
                        **state_diagnostics(model, x, test, test_rows),
                    }
                )
                for comparator, values in ct_metrics(
                    model.world, x, test, test_rows, mean_ct1
                ).items():
                    ct_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "comparator": comparator,
                            "partition": "test",
                            **values,
                        }
                    )
                del model, standalone
        for arm, probabilities, selected_id, logits in predictions:
            path = root / "predictions" / arm / f"seed-{seed}.pt"
            payload = {
                "partition": "test",
                "seed": seed,
                "arm": arm,
                "selected_id": selected_id,
                "patient_ids": test.base.ids,
                "logits": logits,
                "probabilities": probabilities,
                "labels": test.base.labels,
                "threshold": 0.5,
                "protocol_id": object_hash(spec),
            }
            _atomic_torch_save(path, payload)
            for endpoint, metrics in endpoint_metrics(probabilities, test.base.labels).items():
                metrics_rows.append(
                    {"seed": seed, "arm": arm, "endpoint": endpoint, "partition": "test", **metrics}
                )
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "evaluating",
                "test_seeds_completed": spec["seeds"].index(seed) + 1,
                "total_seeds": len(spec["seeds"]),
                "test_scoring_started": True,
            },
        )
        print(f"Test predictions and standalone replay completed: seed={seed}", flush=True)
    report = write_reports(root, metrics_rows, spec, ct_rows, diagnostic_rows)
    atomic_write_private_json(
        root / "verification" / "bundle_replay.json",
        {"status": "passed", "bundles": len(bundle_errors), "results": bundle_errors},
    )
    return report


def run_study(
    project: Path,
    root: Path,
    *,
    smoke: bool = False,
    action: str = "run",
    workers: int = 1,
    worker_seeds: list[int] | None = None,
) -> dict:
    began = time.monotonic()
    pool, spec = prepare(project, root, smoke)
    if action == "prepare":
        return {"status": "prepared", "seeds": len(spec["seeds"])}
    if action == "evaluate":
        return evaluate_tests(pool, root, spec)
    if worker_seeds is not None:
        train_all(pool, root, spec, worker_seeds)
        return {"status": "worker_completed", "seeds": worker_seeds}
    if workers > 1:
        processes = []
        try:
            for index in range(min(workers, len(spec["seeds"]))):
                assignment = spec["seeds"][index::workers]
                command = [
                    sys.executable,
                    "-u",
                    str(project / "scripts/run_surgery_s2.py"),
                    "--mode",
                    "smoke" if smoke else "formal",
                    "--worker-seeds",
                    ",".join(str(seed) for seed in assignment),
                ]
                processes.append(subprocess.Popen(command, cwd=project))
            while any(process.poll() is None for process in processes):
                failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"A training worker failed with exit code {failed[0]}")
                completed = len(list(root.glob("world/seed-*/completed.json")))
                completed += len(list(root.glob("fits/*/seed-*/completed.json")))
                atomic_write_private_json(
                    root / "progress.json",
                    {
                        "status": "training",
                        "completed_phases": completed,
                        "total_phases": len(spec["seeds"]) * 7,
                        "workers": len(processes),
                        "test_scoring_started": False,
                    },
                )
                time.sleep(5)
            if any(process.returncode for process in processes):
                raise RuntimeError("A training worker failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
    else:
        train_all(pool, root, spec)
    if smoke:
        # Use validation patients for software replay. Test patients remain unused.
        seed = spec["seeds"][0]
        dev, x, snapshot, train, validation, anchor = seed_context(pool, root, seed)
        for loss in spec["losses"]:
            for architecture in spec["architectures"]:
                phase = phase_directory(root, seed, architecture, loss)
                saved = torch.load(phase / "validation.pt", weights_only=True, map_location="cpu")
                sample = subset_pool(dev, saved["patient_ids"])
                standalone = predict_bundle(
                    phase / "inference.pt",
                    [sample.base.clinical[p] for p in sample.base.ids],
                    [sample.base.treatments[p] for p in sample.base.ids],
                    sample.base.interval,
                    sample.base.ct0,
                    sample.surgery,
                )
                if not torch.allclose(
                    standalone["logits"], saved["logits"].double(), atol=3e-5, rtol=1e-5
                ):
                    raise ValueError("Smoke standalone inference mismatch")
        report = {
            "status": "smoke_completed",
            "test_evaluated": False,
            "phases": 7,
            "epochs_per_phase": 2,
            "bundle_replays": 6,
        }
    else:
        report = evaluate_tests(pool, root, spec)
    report["seconds"] = time.monotonic() - began
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(
        root / "progress.json", {"status": report["status"], "seconds": report["seconds"]}
    )
    return report
