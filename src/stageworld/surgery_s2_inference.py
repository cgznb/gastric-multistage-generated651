"""Standalone input-only inference with no training-data or target-file dependency."""

from pathlib import Path

import torch

from stageworld.data.baseline_clinical import encode_baseline
from stageworld.data.treatment_compact import encode_compact
from stageworld.surgery_s2_models import SurgeryClassifier
from stageworld.synthetic_workflow import _atomic_torch_save


def export_bundle(
    path: Path, snapshot: dict, model: SurgeryClassifier, selected: dict, protocol_id: str
) -> None:
    if not torch.equal(model.residual_scale.cpu(), torch.ones(2)):
        raise ValueError("Residual coefficient is fixed at one")
    _atomic_torch_save(
        path,
        {
            "schema": "generated651-surgery-s2-inference-v1",
            "inputs": {key: snapshot[key] for key in ("clinical", "support", "mean", "scale")},
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "architecture": model.architecture,
            "image_dim": model.world.input_mean.numel(),
            "threshold": 0.5,
            "ct1_required": False,
            "outcomes_required": False,
            "selected_id": selected["selected_id"],
            "protocol_id": protocol_id,
            "model_selection": "validation_mean_AP",
            "surgery_statuses": [0, 1, 2, 3],
            "clinical_fields": ["sex", "age", "bmi", "ct_stage", "cn_stage", "cm_stage"],
            "attention_fastpath": False,
        },
    )


@torch.inference_mode()
def predict_bundle(
    path: Path,
    clinical: list[dict],
    treatments: list[dict],
    interval: torch.Tensor,
    ct0: torch.Tensor,
    surgery: torch.Tensor,
) -> dict:
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    if (
        bundle["schema"] != "generated651-surgery-s2-inference-v1"
        or bundle["ct1_required"]
        or bundle["outcomes_required"]
        or bundle["threshold"] != 0.5
        or bundle["attention_fastpath"] is not False
    ):
        raise ValueError("Unsupported inference contract")
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
        raise ValueError("Invalid inference inputs")
    torch.backends.mha.set_fastpath_enabled(False)
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
        raise ValueError("Nonfinite transformed input")
    state = bundle["model_state"]
    model = SurgeryClassifier(
        361,
        bundle["architecture"],
        state["anchor_weight"],
        state["anchor_bias"],
        bundle["image_dim"],
    ).eval()
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.residual_scale, torch.ones(2)):
        raise ValueError("Inference residual coefficient differs")
    logits, features, incomplete = [], [], []
    for rows in torch.arange(count).split(32):
        result = model.forward_details(
            x[rows], ct0[rows].float(), torch.ones(len(rows), dtype=torch.bool), surgery[rows]
        )
        logits.append(result["logits"])
        features.append(result["features"])
        incomplete.append(result["history_incomplete"])
    values = torch.cat(logits).double()
    return {
        "logits": values,
        "probabilities": values.sigmoid(),
        "decisions": values.sigmoid() >= 0.5,
        "ct_features": torch.cat(features),
        "history_incomplete": torch.cat(incomplete),
    }
