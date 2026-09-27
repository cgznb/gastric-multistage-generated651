"""Aggregate diagnostics in eval mode, kept distinct from online training losses."""

import numpy as np
import torch
from torch.nn import functional as F

from stageworld.generated700_models import endpoint_loss, feature_set_loss
from stageworld.surgery_s2_evaluation import endpoint_metrics


def balanced_components(logits, labels, valid, weights):
    terms = []
    for i in range(2):
        selected = valid[:, i]
        z, y = logits[selected, i].float(), labels[selected, i].float()
        weight = torch.where(y.bool(), weights[i], 1.0)
        terms.append(
            (F.binary_cross_entropy_with_logits(z, y, reduction="none") * weight).sum()
            / weight.sum()
        )
    return terms


def metrics_from_predictions(output, pool, rows, weights, ct_weight):
    logits, prediction = output["logits"], output["features"]
    labels, valid = pool.base.labels[rows], pool.base.valid[rows]
    if not valid.all():
        raise ValueError("This study requires complete endpoint labels")
    ct_valid = (pool.base.ct0_valid & pool.base.ct1_valid)[rows]
    ct_loss = float(feature_set_loss(prediction, pool.base.ct1_tokens[rows], ct_valid))
    terms = balanced_components(logits, labels, valid, weights.cpu())
    classification = float(endpoint_loss(logits, labels, valid, weights.cpu()))
    endpoints = endpoint_metrics(logits.double().sigmoid(), labels)
    result = {
        "ct_loss": ct_loss,
        "pcr_balanced_bce": float(terms[0]),
        "recurrence_balanced_bce": float(terms[1]),
        "joint_loss": classification + ct_weight * ct_loss,
        "mean_AP": float(np.mean([endpoints[e]["auprc"] for e in ("pcr", "recurrence")])),
        "s2_relative_update_mean": float(output["s2_ratio"].mean()),
    }
    for i, endpoint in enumerate(("pcr", "recurrence")):
        for metric in ("auroc", "auprc", "bce", "brier", "sensitivity", "specificity"):
            result[f"{endpoint}_{metric}"] = endpoints[endpoint][metric]
        result[f"{endpoint}_raw_residual_near_bound"] = float(
            (output["delta"][:, i].abs() >= 1.8).float().mean()
        )
    return result, endpoints


@torch.inference_mode()
def predict(model, x, pool, rows):
    model.eval()
    device = next(model.parameters()).device
    result = {key: [] for key in ("logits", "features", "delta", "s2_ratio")}
    for batch in rows.split(32):
        out = model.forward_details(
            x[batch].to(device),
            pool.base.ct0[batch].to(device),
            pool.base.ct0_valid[batch].to(device),
            pool.surgery[batch].to(device),
        )
        for key in ("logits", "features", "delta"):
            result[key].append(out[key].detach().float().cpu())
        numerator = (out["s2"] - out["s1"]).flatten(1).norm(dim=1)
        denominator = out["s1"].flatten(1).norm(dim=1).clamp_min(1e-8)
        result["s2_ratio"].append((numerator / denominator).float().cpu())
    joined = {key: torch.cat(value) for key, value in result.items()}
    if any(not value.isfinite().all() for value in joined.values()):
        raise ValueError("Nonfinite evaluation output")
    return joined


def evaluate(model, x, pool, rows, weights, ct_weight):
    return metrics_from_predictions(predict(model, x, pool, rows), pool, rows, weights, ct_weight)[
        0
    ]
