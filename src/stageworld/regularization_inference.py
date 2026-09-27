"""Standalone inference carrying model settings and frozen input transforms."""

from pathlib import Path

import torch

from stageworld.data.baseline_clinical import encode_baseline
from stageworld.data.treatment_compact import encode_compact
from stageworld.regularization_models import build_model
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(path: Path, snapshot, model, config, report, protocol_id):
    _atomic_torch_save(
        path,
        {
            "schema": "generated651-regularization-inference-v1",
            "inputs": {key: snapshot[key] for key in ("clinical", "support", "mean", "scale")},
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "seed": report["seed"],
            "image_dim": model.world.input_mean.numel(),
            "threshold": 0.5,
            "ct1_required": False,
            "outcomes_required": False,
            "selected_id": report["selected_id"],
            "protocol_id": protocol_id,
        },
    )


@torch.inference_mode()
def predict_bundle(path, clinical, treatments, interval, ct0, surgery):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    if (
        bundle["schema"] != "generated651-regularization-inference-v1"
        or bundle["ct1_required"]
        or bundle["outcomes_required"]
        or bundle["threshold"] != 0.5
    ):
        raise ValueError("Unsupported input-only inference contract")
    count = len(clinical)
    if (
        not count
        or len(treatments) != count
        or interval.shape != (count,)
        or ct0.shape != (count, 27, bundle["image_dim"])
        or surgery.shape != (count,)
        or surgery.dtype != torch.long
        or not ((surgery >= 0) & (surgery < 4)).all()
        or not torch.isfinite(ct0).all()
        or not torch.isfinite(interval).all()
        or not (interval > 0).all()
    ):
        raise ValueError("Invalid standalone inference inputs")
    snapshot = bundle["inputs"]
    baseline = encode_baseline(clinical, snapshot["clinical"])
    treatment, _ = encode_compact(treatments, snapshot["support"])
    raw = torch.cat(
        (
            baseline.ridge_features(),
            treatment.flatten(1),
            torch.log1p(interval[:, None].float() / 30),
        ),
        1,
    ).double()
    x = ((raw - snapshot["mean"]) / snapshot["scale"]).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite frozen preprocessing result")
    state = bundle["model_state"]
    anchor = {"weight": state["anchor_weight"], "bias": state["anchor_bias"]}
    model = build_model(bundle["config"], bundle["image_dim"], bundle["seed"], anchor).eval()
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.residual_scale, torch.full((2,), bundle["config"]["alpha"])):
        raise ValueError("Inference residual coefficient mismatch")
    logits = []
    for rows in torch.arange(count).split(32):
        logits.append(
            model(
                x[rows], ct0[rows].float(), torch.ones(len(rows), dtype=torch.bool), surgery[rows]
            )
        )
    values = torch.cat(logits).double()
    return {
        "logits": values,
        "probabilities": values.sigmoid(),
        "decisions": values.sigmoid() >= 0.5,
    }
