"""Validation-selected two-phase training with exact epoch-boundary recovery."""

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary700_statistics import positive_weights
from stageworld.ct6_training import EarlyStopState
from stageworld.generated700_models import FutureWorld, endpoint_loss, feature_set_loss
from stageworld.generated_workflow import _seed
from stageworld.surgery_s2_data import SurgeryPool
from stageworld.surgery_s2_io import bound_json, checkpoint_link, object_hash
from stageworld.surgery_s2_models import SurgeryClassifier
from stageworld.synthetic_workflow import _atomic_torch_save
from stageworld.training import capture_rng_state, restore_rng_state


def build_model(
    architecture: str,
    image_dim: int,
    seed: int,
    anchor: dict | None = None,
    parent: dict | None = None,
):
    torch.backends.mha.set_fastpath_enabled(False)
    _seed(seed)
    if architecture == "world":
        return FutureWorld(361, image_dim)
    if anchor is None or parent is None:
        raise ValueError("Joint training requires a matching anchor and CT pretrain")
    model = SurgeryClassifier(
        361, architecture, anchor["weight"], anchor["bias"], image_dim=image_dim, seed=seed
    )
    model.world.load_state_dict(parent["model_state"], strict=True)
    return model


@torch.inference_mode()
def infer(
    model, x: torch.Tensor, pool: SurgeryPool, rows: torch.Tensor, *, world: bool = False
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    result = []
    for batch in rows.split(32):
        args = x[batch].to(device), pool.base.ct0[batch].to(device)
        values = (
            model(*args)[2]
            if world
            else model(*args, pool.base.ct0_valid[batch].to(device), pool.surgery[batch].to(device))
        )
        result.append(values.float().cpu())
    values = torch.cat(result)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite evaluation output")
    return values


def score(
    model, x: torch.Tensor, pool: SurgeryPool, rows: torch.Tensor, world: bool
) -> tuple[float, dict]:
    result = infer(model, x, pool, rows, world=world)
    if world:
        mask = (pool.base.ct0_valid & pool.base.ct1_valid)[rows]
        loss = float(feature_set_loss(result, pool.base.ct1_tokens[rows], mask))
        return loss, {"ct_feature_set_loss": loss}
    probabilities = result.double().sigmoid().numpy()
    ap = [
        float(average_precision_score(pool.base.labels[rows, i].numpy(), probabilities[:, i]))
        for i in range(2)
    ]
    return -float(np.mean(ap)), {"pcr_AP": ap[0], "recurrence_AP": ap[1]}


def train_phase(
    pool: SurgeryPool,
    x: torch.Tensor,
    snapshot: dict,
    train: torch.Tensor,
    validation: torch.Tensor,
    root: Path,
    *,
    seed: int,
    architecture: str,
    loss_name: str,
    spec: dict,
    anchor: dict | None = None,
    parent_path: Path | None = None,
    device: str | None = None,
    interrupt_after_update: int | None = None,
) -> tuple[torch.nn.Module, dict]:
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
        raise ValueError("Trainer accepts only the isolated train/validation pool")
    world = architecture == "world"
    if architecture not in ("world", *spec["architectures"]) or loss_name not in spec["losses"]:
        raise ValueError("Unregistered training arm")
    parent = (
        None
        if parent_path is None
        else torch.load(parent_path, weights_only=True, map_location="cpu")
    )
    if not world:
        if parent is None or anchor is None:
            raise ValueError("Missing training-only initialization")
        expected = {
            "snapshot_id": snapshot["artifact_id"],
            "seed": seed,
            "architecture": "world",
            "train_ids": train_ids,
            "validation_ids": val_ids,
            "protocol_id": object_hash(spec),
        }
        if any(parent["contract"].get(k) != v for k, v in expected.items()):
            raise ValueError("CT pretrain parent belongs to a different partition/protocol")
        if anchor["contract"]["snapshot_id"] != snapshot["artifact_id"]:
            raise ValueError("Anchor belongs to a different train partition")
    contract = {
        "schema": "surgery-s2-phase-v1",
        "protocol_id": object_hash(spec),
        "architecture": architecture,
        "loss": loss_name,
        "seed": seed,
        "snapshot_id": snapshot["artifact_id"],
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "epochs": spec["phase_epochs"],
        "parent_id": None if parent is None else parent["artifact_id"],
        "anchor_id": None if anchor is None else anchor["artifact_id"],
        "image_dim": base.ct0.shape[-1],
        "tabular_dim": x.shape[-1],
        "device_type": device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }
    bound_json(root / "contract.json", contract)
    model = build_model(architecture, base.ct0.shape[-1], seed, anchor, parent)
    target = torch.device(contract["device_type"])
    model = model.to(target)
    if (root / "completed.json").exists():
        saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Completed checkpoint binding differs")
        model.load_state_dict(saved["model_state"], strict=True)
        return model.eval(), read_json(root / "completed.json")
    if world:
        usable = train[(base.ct0_valid & base.ct1_valid)[train]]
        if not len(usable):
            raise ValueError("No paired CT supervision")
        model.fit_statistics(base.ct0[usable].to(target), base.ct1_tokens[usable].to(target))
        groups = [{"params": list(model.parameters()), "lr": spec["world_learning_rate"]}]
        weights = torch.ones(2, device=target)
        fitting = usable
    else:
        groups = [
            {"params": list(model.world.parameters()), "lr": spec["joint_world_learning_rate"]},
            {
                "params": list(model.endpoints.parameters()),
                "lr": spec["joint_head_surgery_learning_rate"],
            },
        ]
        if model.surgery is not None:
            groups.append(
                {
                    "params": list(model.surgery.parameters()),
                    "lr": spec["joint_head_surgery_learning_rate"],
                }
            )
        weights = positive_weights(
            base.labels[train], base.valid[train], "none" if loss_name == "bce" else "balanced"
        ).to(target)
        fitting = train
    optimizer = torch.optim.AdamW(groups, weight_decay=spec["weight_decay"])
    stopper, history, updates, first, elapsed = EarlyStopState(), [], 0, 1, 0.0
    if (root / "latest.pt").exists():
        saved = torch.load(root / "latest.pt", weights_only=True, map_location="cpu")
        if saved["contract"] != contract:
            raise ValueError("Recovery contract differs")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng_state(saved["rng_state"])
        stopper = EarlyStopState(**saved["early_stop"])
        history, updates, first = saved["history"], saved["updates"], saved["epoch"] + 1
        elapsed = saved["elapsed_seconds"]
        if stopper.selected_epoch == saved["epoch"]:
            checkpoint_link(root / "latest.pt", root / "selected.pt")
    tensors = {
        "x": x.to(target),
        "ct0": base.ct0.to(target),
        "ct1": base.ct1_tokens.to(target),
        "ct0_valid": base.ct0_valid.to(target),
        "ct1_valid": (base.ct0_valid & base.ct1_valid).to(target),
        "labels": base.labels.to(target),
        "valid": base.valid.to(target),
        "surgery": pool.surgery.to(target),
    }
    trainable = list(model.parameters())
    began = time.monotonic()
    for epoch in range(first, spec["phase_epochs"] + 1):
        if stopper.stale_epochs >= spec["patience"]:
            break
        model.train()
        generator = torch.Generator().manual_seed(seed * 100003 + epoch)
        order = fitting[torch.randperm(len(fitting), generator=generator)]
        gradient_max, surgery_gradient_max, total_loss = 0.0, 0.0, 0.0
        for rows in order.split(spec["batch_size"]):
            batch = {name: value[rows] for name, value in tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(target.type, dtype=torch.bfloat16, enabled=target.type == "cuda"):
                if world:
                    future = model(batch["x"], batch["ct0"])[2]
                    loss = feature_set_loss(future, batch["ct1"], batch["ct1_valid"])
                else:
                    logits, future = model.forward_with_features(
                        batch["x"], batch["ct0"], batch["ct0_valid"], batch["surgery"]
                    )
                    loss = endpoint_loss(
                        logits,
                        batch["labels"],
                        batch["valid"],
                        weights,
                        "focal" if loss_name == "focal" else "bce",
                    )
                    loss = loss + spec["ct_loss_coefficient"] * feature_set_loss(
                        future, batch["ct1"], batch["ct1_valid"]
                    )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization loss")
            loss.backward()
            if not world and model.surgery is not None:
                gradients = [p.grad for p in model.surgery.parameters() if p.grad is not None]
                norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
                surgery_gradient_max = max(surgery_gradient_max, float(norm))
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, spec["gradient_clip"], error_if_nonfinite=True
            )
            gradient_max = max(gradient_max, float(norm))
            total_loss += float(loss.detach()) * len(rows)
            optimizer.step()
            updates += 1
            if interrupt_after_update == updates:
                raise RuntimeError("intentional_partial_epoch_interruption")
        value, metrics = score(model, x, pool, validation, world)
        selected = stopper.update(value, epoch, spec["min_delta"])
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "score": value,
                "metrics": metrics,
                "training_loss": total_loss / len(fitting),
                "selected": selected,
                "gradient_norm_max": gradient_max,
                "surgery_gradient_norm_max": surgery_gradient_max,
            }
        )
        payload = {
            "artifact_id": new_artifact_id("surgery-s2-checkpoint"),
            "contract": contract,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": epoch,
            "updates": updates,
            "early_stop": asdict(stopper),
            "history": history,
            "positive_weights": weights.detach().cpu(),
            "elapsed_seconds": elapsed + time.monotonic() - began,
        }
        _atomic_torch_save(root / "latest.pt", payload)
        if selected:
            checkpoint_link(root / "latest.pt", root / "selected.pt")
        atomic_write_private_json(
            root / "progress.json",
            {
                "status": "running",
                "seed": seed,
                "architecture": architecture,
                "loss": loss_name,
                "epoch": epoch,
                "updates": updates,
                "selected_epoch": stopper.selected_epoch,
                "seconds": payload["elapsed_seconds"],
            },
        )
    checkpoint_link(root / "latest.pt", root / "final.pt")
    saved = torch.load(root / "selected.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model_state"], strict=True)
    actual, _ = score(model, x, pool, validation, world)
    if not math.isclose(actual, saved["history"][-1]["score"], abs_tol=1e-7, rel_tol=1e-6):
        raise ValueError("Selected checkpoint score does not replay")
    if not world:
        if (
            not torch.equal(model.anchor_weight.cpu(), anchor["weight"])
            or not torch.equal(model.anchor_bias.cpu(), anchor["bias"])
            or not torch.equal(model.residual_scale.cpu(), torch.ones(2))
        ):
            raise ValueError("Frozen anchor or fixed residual coefficient changed")
    report = {
        "status": "completed",
        "seed": seed,
        "architecture": architecture,
        "loss": loss_name,
        "completed_epochs": history[-1]["epoch"],
        "selected_epoch": saved["epoch"],
        "selected_score": actual,
        "selected_id": saved["artifact_id"],
        "updates": updates,
        "seconds": elapsed + time.monotonic() - began,
        "parameters": sum(p.numel() for p in model.parameters()),
        "positive_weights": weights.cpu().tolist(),
        "checkpoint_replay_passed": True,
        "world_jointly_adapted": not world,
        "surgery_gradient_norm_max": max(h["surgery_gradient_norm_max"] for h in history),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if target.type == "cuda"
        else 0,
    }
    atomic_write_private_json(root / "completed.json", report)
    atomic_write_private_json(root / "progress.json", report)
    print(
        f"Completed seed={seed} {architecture}/{loss_name}: "
        f"epochs={report['completed_epochs']} selected={saved['epoch']} "
        f"seconds={report['seconds']:.1f}",
        flush=True,
    )
    return model.eval(), report
