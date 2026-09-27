"""Restartable joint training with complete development diagnostics and compact retention."""

import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary700_statistics import positive_weights
from stageworld.ct6_training import EarlyStopState
from stageworld.generated700_models import FutureWorld, endpoint_loss, feature_set_loss
from stageworld.regularization_metrics import balanced_components, evaluate
from stageworld.regularization_models import build_model
from stageworld.surgery_s2_io import bound_json, object_hash
from stageworld.surgery_s2_training import score
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state


def compact(payload):
    return {k: v for k, v in payload.items() if k not in ("optimizer_state", "rng_state")}


def train_world(
    pool,
    x,
    snapshot,
    train,
    validation,
    root: Path,
    *,
    seed,
    batch_size,
    spec,
    device="cuda",
):
    """Stage-one world pretraining from random initialization; no reused weights."""
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = pool.base
    train_ids = [base.ids[i] for i in train.tolist()]
    val_ids = [base.ids[i] for i in validation.tolist()]
    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) != set(snapshot["fit_ids"])
        or set(train_ids) | set(val_ids) != set(base.ids)
        or len(set(base.ids)) != len(base.ids)
        or not len(validation)
    ):
        raise ValueError("Trainer requires physically isolated train/validation patients")
    contract = {
        "schema": "fullretrain-world-v1",
        "protocol_id": object_hash(spec),
        "architecture": "world",
        "seed": seed,
        "batch_size": batch_size,
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "image_dim": base.ct0.shape[-1],
        "tabular_dim": x.shape[-1],
        "device_type": device,
    }
    bound_json(root / "contract.json", contract)
    model = FutureWorld(361, base.ct0.shape[-1]).to(device)
    if (root / "completed.json").exists():
        saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Completed world checkpoint binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        return model.eval(), read_json(root / "completed.json")
    usable = train[(base.ct0_valid & base.ct1_valid)[train]]
    if not len(usable):
        raise ValueError("No paired CT supervision")
    model.fit_statistics(base.ct0[usable].to(device), base.ct1_tokens[usable].to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["world_learning_rate"], weight_decay=spec["weight_decay"]
    )
    stopper, history, updates, first, elapsed = EarlyStopState(), [], 0, 1, 0.0
    initial = None
    if (root / "recovery.pt").exists():
        saved = torch.load(root / "recovery.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Recovery contract differs")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates, first = saved["history"], saved["updates"], saved["epoch"] + 1
        elapsed = saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            _atomic_torch_save(root / "selected.pt", compact(saved))
        restore_rng_state(saved["rng_state"])
    if initial is None:
        initial = {
            name: score(model, x, pool, rows, True)[1]
            for name, rows in (("train", train), ("validation", validation))
        }
    tensors = {
        "x": x.to(device),
        "ct0": base.ct0.to(device),
        "ct1": base.ct1_tokens.to(device),
        "ct0_valid": base.ct0_valid.to(device),
        "ct1_valid": (base.ct0_valid & base.ct1_valid).to(device),
    }
    began = time.monotonic()
    for epoch in range(first, spec["phase_epochs"] + 1):
        if stopper.stale_epochs >= spec["patience"]:
            break
        model.train()
        generator = torch.Generator().manual_seed(seed * 100003 + epoch)
        order = usable[torch.randperm(len(usable), generator=generator)]
        norms = []
        for rows in order.split(batch_size):
            batch = {name: value[rows] for name, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=device == "cuda"):
                future = model(batch["x"], batch["ct0"])[2]
                loss = feature_set_loss(future, batch["ct1"], batch["ct1_valid"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization loss")
            loss.backward()
            norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()), spec["gradient_clip"], error_if_nonfinite=True
                    )
                )
            )
            optimizer.step()
            updates += 1
        diagnostics = {
            name: score(model, x, pool, rows, True)[1]
            for name, rows in (("train", train), ("validation", validation))
        }
        validation_loss = diagnostics["validation"]["ct_feature_set_loss"]
        selected = stopper.update(validation_loss, epoch, spec["min_delta"])
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": validation_loss,
                "selected": selected,
                **diagnostics,
                "gradient_norm_max": max(norms),
                "clipped_batches": sum(n > spec["gradient_clip"] for n in norms),
                "batches": len(norms),
                "clip_fraction": sum(n > spec["gradient_clip"] for n in norms) / len(norms),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        payload = {
            "artifact_id": new_artifact_id("fullretrain-world-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "initial_metrics": initial,
            "history": history,
            "elapsed_seconds": elapsed + time.monotonic() - began,
        }
        _atomic_torch_save(root / "recovery.pt", payload)
        if selected:
            _atomic_torch_save(root / "selected.pt", compact(payload))
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "training",
                "seed": seed,
                "batch_size": batch_size,
                "epoch": epoch,
                "selected_epoch": stopper.selected_epoch,
                "seconds": payload["elapsed_seconds"],
                "validation_ct_loss": validation_loss,
                "test_scoring_started": False,
            },
        )
    final = torch.load(root / "recovery.pt", weights_only=True, map_location="cpu")
    _atomic_torch_save(root / "final.pt", compact(final))
    saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model_state"], strict=True)
    actual = score(model, x, pool, validation, True)[1]
    if not math.isclose(actual["ct_feature_set_loss"], saved["history"][-1]["score"], abs_tol=1e-7):
        raise ValueError("Selected CT loss did not replay")
    report = {
        "status": "completed",
        "seed": seed,
        "batch_size": batch_size,
        "completed_epochs": len(history),
        "selected_epoch": saved["epoch"],
        "selected_id": saved["artifact_id"],
        "validation": actual,
        "updates": updates,
        "seconds": elapsed + time.monotonic() - began,
        "parameters": sum(p.numel() for p in model.parameters()),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }
    atomic_write_private_json(root / "history.json", {"initial": initial, "epochs": history})
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(root / "progress.json", report)
    (root / "recovery.pt").unlink()
    print(
        f"Completed world seed={seed} batch={batch_size}: epochs={len(history)} "
        f"selected={saved['epoch']} seconds={report['seconds']:.1f}",
        flush=True,
    )
    return model.eval(), report


def train_fit(
    pool,
    x,
    snapshot,
    train,
    validation,
    root: Path,
    *,
    seed,
    config,
    spec,
    anchor,
    parent,
    device="cuda",
    interrupt_after_update=None,
):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = pool.base
    train_ids = [base.ids[i] for i in train.tolist()]
    val_ids = [base.ids[i] for i in validation.tolist()]
    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) != set(snapshot["fit_ids"])
        or set(train_ids) | set(val_ids) != set(base.ids)
        or len(set(base.ids)) != len(base.ids)
        or not len(validation)
    ):
        raise ValueError("Trainer requires physically isolated train/validation patients")
    expected = {
        "seed": seed,
        "architecture": "world",
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "protocol_id": object_hash(spec),
    }
    if any(parent["contract"].get(k) != v for k, v in expected.items()):
        raise ValueError("Pretrained parent partition/protocol mismatch")
    if anchor["contract"]["snapshot_id"] != snapshot["artifact_id"]:
        raise ValueError("Frozen anchor partition mismatch")
    if config not in spec["arms"] or seed not in spec["seeds"]:
        raise ValueError("Unregistered arm or seed")
    contract = {
        "schema": "generated651-regularization-fit-v1",
        "protocol_id": object_hash(spec),
        "seed": seed,
        "arm": config,
        "snapshot_id": snapshot["artifact_id"],
        "parent_id": parent["artifact_id"],
        "anchor_id": anchor["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "image_dim": base.ct0.shape[-1],
        "device": device,
    }
    bound_json(root / "contract.json", contract)
    model = build_model(config, base.ct0.shape[-1], seed, anchor, parent).to(device)
    if (root / "completed.json").exists():
        saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Completed fit binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        return model.eval(), read_json(root / "completed.json")
    groups = [
        {"params": list(model.world.parameters()), "lr": config["world_lr"]},
        {"params": list(model.endpoints.parameters()), "lr": config["head_surgery_lr"]},
        {"params": list(model.surgery.parameters()), "lr": config["head_surgery_lr"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    weights = positive_weights(base.labels[train], base.valid[train], "balanced").to(device)
    stopper, history, updates, first, elapsed = EarlyStopState(), [], 0, 1, 0.0
    initial = None
    if (root / "recovery.pt").exists():
        saved = torch.load(root / "recovery.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Recovery binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates, first = saved["history"], saved["updates"], saved["epoch"] + 1
        initial, elapsed = saved["initial_metrics"], saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            _atomic_torch_save(root / "selected.pt", compact(saved))
        restore_rng_state(saved["rng_state"])
    if initial is None:
        initial = {
            name: evaluate(model, x, pool, rows, weights, config["ct_weight"])
            for name, rows in (("train", train), ("validation", validation))
        }
    tensors = {
        "x": x.to(device),
        "ct0": base.ct0.to(device),
        "ct1": base.ct1_tokens.to(device),
        "ct0_valid": base.ct0_valid.to(device),
        "ct1_valid": (base.ct0_valid & base.ct1_valid).to(device),
        "labels": base.labels.to(device),
        "valid": base.valid.to(device),
        "surgery": pool.surgery.to(device),
    }
    trainable = list(model.parameters())
    began = time.monotonic()
    for epoch in range(first, spec["phase_epochs"] + 1):
        if stopper.stale_epochs >= spec["patience"]:
            break
        if shutil.disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError("Less than 5 GiB free; recovery retained, no files removed")
        model.train()
        generator = torch.Generator().manual_seed(seed * 100003 + epoch)
        order = train[torch.randperm(len(train), generator=generator)]
        norms, surgery_norms = [], []
        totals = {
            k: 0.0 for k in ("joint_loss", "pcr_balanced_bce", "recurrence_balanced_bce", "ct_loss")
        }
        for rows in order.split(config["batch_size"]):
            batch = {name: value[rows] for name, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                torch.device(device).type, dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                logits, future = model.forward_with_features(
                    batch["x"], batch["ct0"], batch["ct0_valid"], batch["surgery"]
                )
                loss = endpoint_loss(logits, batch["labels"], batch["valid"], weights, "bce")
                ct_loss = feature_set_loss(future, batch["ct1"], batch["ct1_valid"])
                loss = loss + config["ct_weight"] * ct_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization loss")
            loss.backward()
            gradients = [p.grad for p in model.surgery.parameters() if p.grad is not None]
            surgery_norms.append(
                float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients))))
            )
            norms.append(
                float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable, spec["gradient_clip"], error_if_nonfinite=True
                    )
                )
            )
            with torch.no_grad():
                terms = balanced_components(logits, batch["labels"], batch["valid"], weights)
                for key, value in zip(totals, (loss, terms[0], terms[1], ct_loss), strict=True):
                    totals[key] += float(value) * len(rows)
            optimizer.step()
            updates += 1
            if interrupt_after_update == updates:
                raise RuntimeError("intentional_partial_epoch_interruption")
        diagnostics = {
            name: evaluate(model, x, pool, rows, weights, config["ct_weight"])
            for name, rows in (("train", train), ("validation", validation))
        }
        score = -diagnostics["validation"]["mean_AP"]
        selected = stopper.update(score, epoch, spec["min_delta"])
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": score,
                "selected": selected,
                "online": {k: v / len(train) for k, v in totals.items()},
                **diagnostics,
                "gradient_norm_max": max(norms),
                "gradient_norm_median": float(np.median(norms)),
                "gradient_norm_p90": float(np.quantile(norms, 0.9)),
                "clipped_batches": sum(n > spec["gradient_clip"] for n in norms),
                "batches": len(norms),
                "clip_fraction": sum(n > spec["gradient_clip"] for n in norms) / len(norms),
                "surgery_gradient_norm_max": max(surgery_norms),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        payload = {
            "artifact_id": new_artifact_id("regularization-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "initial_metrics": initial,
            "history": history,
            "positive_weights": weights.cpu(),
            "elapsed_seconds": elapsed + time.monotonic() - began,
        }
        _atomic_torch_save(root / "recovery.pt", payload)
        if selected:
            _atomic_torch_save(root / "selected.pt", compact(payload))
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "training",
                "seed": seed,
                "arm": config["name"],
                "epoch": epoch,
                "selected_epoch": stopper.selected_epoch,
                "seconds": payload["elapsed_seconds"],
                "validation_mean_AP": -score,
                "test_scoring_started": False,
            },
        )
    final = torch.load(root / "recovery.pt", weights_only=True, map_location="cpu")
    _atomic_torch_save(root / "final.pt", compact(final))
    saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model_state"], strict=True)
    actual = evaluate(model, x, pool, validation, weights, config["ct_weight"])
    if not math.isclose(-actual["mean_AP"], saved["history"][-1]["score"], abs_tol=1e-7):
        raise ValueError("Selected AP did not replay")
    if (
        not torch.equal(model.anchor_weight.cpu(), anchor["weight"])
        or not torch.equal(model.anchor_bias.cpu(), anchor["bias"])
        or not torch.equal(model.residual_scale.cpu(), torch.full((2,), config["alpha"]))
    ):
        raise ValueError("Frozen anchor or fixed residual scale changed")
    report = {
        "status": "completed",
        "seed": seed,
        "arm": config["name"],
        "completed_epochs": len(history),
        "selected_epoch": saved["epoch"],
        "selected_id": saved["artifact_id"],
        "validation": actual,
        "selected_train": saved["history"][-1]["train"],
        "updates": updates,
        "seconds": elapsed + time.monotonic() - began,
        "parameters": sum(p.numel() for p in model.parameters()),
        "surgery_gradient_norm_max": max(h["surgery_gradient_norm_max"] for h in history),
        "world_jointly_adapted": any(
            not torch.equal(v.cpu(), parent["model_state"][k])
            for k, v in model.world.state_dict().items()
        ),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
    }
    atomic_write_private_json(root / "history.json", {"initial": initial, "epochs": history})
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(root / "progress.json", report)
    (root / "recovery.pt").unlink()
    print(
        f"Completed seed={seed} arm={config['name']}: epochs={len(history)} "
        f"selected={saved['epoch']} seconds={report['seconds']:.1f}",
        flush=True,
    )
    return model.eval(), report
