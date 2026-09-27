"""Fixed-threshold endpoint metrics and complete, unfiltered holdout reports."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stageworld.artifacts import atomic_write_private_json
from stageworld.generated651_evaluation import write_csv
from stageworld.generated700_evaluation import points
from stageworld.generated700_models import feature_set_loss
from stageworld.surgery_s2_data import SurgeryPool
from stageworld.surgery_s2_training import infer


def endpoint_metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict:
    result = {}
    for i, endpoint in enumerate(("pcr", "recurrence")):
        p = probabilities[:, i].double().numpy()
        y = labels[:, i].numpy()
        metrics = points(p, y, p >= 0.5)
        tp, fp, tn, fn = (metrics[k] for k in ("tp", "fp", "tn", "fn"))
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics.update(
            {
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "npv": tn / (tn + fn) if tn + fn else None,
                "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
            }
        )
        result[endpoint] = metrics
    return result


@torch.inference_mode()
def state_diagnostics(model, x: torch.Tensor, pool: SurgeryPool, rows: torch.Tensor) -> dict:
    model.eval()
    device = next(model.parameters()).device
    states, ratios, residuals = [], [], []
    for batch in rows.split(32):
        result = model.forward_details(
            x[batch].to(device),
            pool.base.ct0[batch].to(device),
            pool.base.ct0_valid[batch].to(device),
            pool.surgery[batch].to(device),
        )
        s1, s2 = result["s1"].float(), result["s2"].float()
        ratio = (s2 - s1).flatten(1).norm(dim=1) / s1.flatten(1).norm(dim=1).clamp_min(1e-8)
        ratios.append(ratio.cpu())
        states.append(s2.cpu())
        residuals.append(result["delta"][:, 1].cpu())
    ratio, state, residual = torch.cat(ratios), torch.cat(states), torch.cat(residuals)
    return {
        "s2_relative_update_mean": float(ratio.mean()),
        "s2_relative_update_max": float(ratio.max()),
        "s2_patient_variance_mean": float(state.var(0, unbiased=False).mean()),
        "recurrence_residual_mean": float(residual.mean()),
        "recurrence_residual_sd": float(residual.std(unbiased=False)),
        "recurrence_residual_min": float(residual.min()),
        "recurrence_residual_max": float(residual.max()),
    }


@torch.inference_mode()
def ct_metrics(
    world, x: torch.Tensor, pool: SurgeryPool, rows: torch.Tensor, train_mean_tokens: torch.Tensor
) -> dict:
    prediction = infer(world, x, pool, rows, world=True)
    target = pool.base.ct1_tokens[rows]
    values = {
        "generated": prediction,
        "copy_ct0": pool.base.ct0[rows],
        "training_ct1_mean": train_mean_tokens[None].expand_as(target),
    }
    result = {}
    for name, value in values.items():
        p, y = value.mean(1), target.mean(1)
        result[name] = {
            "n": len(rows),
            "global_mse": float((p - y).square().mean()),
            "global_cosine": float(F.cosine_similarity(p, y, dim=1).mean()),
            "feature_set_loss": float(
                feature_set_loss(value, target, torch.ones(len(rows), dtype=torch.bool))
            ),
        }
    return result


def write_reports(
    root: Path, rows: list[dict], spec: dict, ct_rows: list[dict], diagnostics: list[dict]
) -> dict:
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected = len(spec["seeds"]) * (len(spec["architectures"]) * len(spec["losses"]) + 1) * 2
    keys = {(r["seed"], r["arm"], r["endpoint"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise ValueError("Per-seed test metrics are incomplete or repeated")
    write_csv(output / "per_seed_test_metrics.csv", rows)
    write_csv(output / "ct_metrics.csv", ct_rows)
    write_csv(output / "state_diagnostics.csv", diagnostics)
    lookup = {(r["seed"], r["arm"], r["endpoint"]): r for r in rows}
    paired = []
    for seed in spec["seeds"]:
        for loss in spec["losses"]:
            for endpoint in ("pcr", "recurrence"):
                g1, g2 = (lookup[seed, f"{arch}_{loss}", endpoint] for arch in ("g1", "g2"))
                paired.append(
                    {
                        "seed": seed,
                        "loss": loss,
                        "endpoint": endpoint,
                        **{
                            f"delta_{metric}": g2[metric] - g1[metric]
                            for metric in (
                                "auroc",
                                "auprc",
                                "accuracy",
                                "sensitivity",
                                "specificity",
                                "brier",
                            )
                        },
                    }
                )
    write_csv(output / "paired_differences.csv", paired)
    arms = ["logistic"] + [
        f"{arch}_{loss}" for loss in spec["losses"] for arch in spec["architectures"]
    ]
    summary = []
    for arm in arms:
        for endpoint in ("pcr", "recurrence"):
            for metric in (
                "auroc",
                "auprc",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "bce",
                "brier",
                "mcc",
            ):
                values = [lookup[seed, arm, endpoint][metric] for seed in spec["seeds"]]
                values = [value for value in values if value is not None]
                summary.append(
                    {
                        "arm": arm,
                        "endpoint": endpoint,
                        "metric": metric,
                        "mean": float(np.mean(values)) if values else None,
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "defined_seeds": len(values),
                    }
                )
    write_csv(output / "seed_summary.csv", summary)
    wide = []
    for seed in spec["seeds"]:
        row = {"seed": seed, "train_n": 521, "validation_n": 65, "test_n": 65}
        for endpoint in ("pcr", "recurrence"):
            metrics = lookup[seed, "g2_balanced", endpoint]
            row.update(
                {
                    f"{endpoint}_{key}": value
                    for key, value in metrics.items()
                    if key not in ("arm", "seed", "endpoint", "partition")
                }
            )
        wide.append(row)
    write_csv(output / "primary_g2_balanced_per_seed.csv", wide)
    text = [
        "# Generated651 surgery S2: twenty-seed internal test results",
        "",
        "Each seed has a patient-level joint-stratified 521/65/65 split. All 120 joint",
        "models were selected using validation mean endpoint AP before any test scoring.",
        "Threshold = 0.5; neural residual coefficient = 1; no calibration.",
        "",
        "Primary arm: G2 balanced BCE. G1 balanced BCE is the paired reference.",
        "",
        "| Seed | pCR AUROC | pCR AP | pCR recall | Recurrence AUROC | "
        "Recurrence AP | Recurrence recall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in wide:
        values = [
            row[f"{endpoint}_{metric}"]
            for endpoint in ("pcr", "recurrence")
            for metric in ("auroc", "auprc", "sensitivity")
        ]
        text.append(f"| {row['seed']} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    text += [
        "",
        "## All-arm averages",
        "",
        "| Arm | pCR AUROC | pCR AP | Recurrence AUROC | Recurrence AP |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        values = [
            next(
                r
                for r in summary
                if r["arm"] == arm and r["endpoint"] == endpoint and r["metric"] == metric
            )
            for endpoint in ("pcr", "recurrence")
            for metric in ("auroc", "auprc")
        ]
        text.append(
            f"| {arm} | " + " | ".join(f"{r['mean']:.4f} +/- {r['sd']:.4f}" for r in values) + " |"
        )
    text += [
        "",
        "## Interpretation",
        "",
        "SD describes variation across different overlapping patient splits and training seeds.",
        "It is not a confidence interval from 20 independent cohorts. Each test set has only",
        "65 patients. This cohort has been reused in historical development experiments.",
        "All patients have surgery present; S2 tests representation/capacity and cannot",
        "identify the causal effect of surgery. The recurrence endpoint is recorded status,",
        "not prospective survival. Undefined precision/MCC values remain empty in CSV.",
        "",
        "Complete metrics, confusion matrices, all six arms, Logistic, CT diagnostics and",
        "paired differences are in the adjacent CSV files. No seed or loss is omitted.",
        "",
    ]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    result = {
        "status": "completed",
        "seed_count": len(spec["seeds"]),
        "test_metric_rows": len(rows),
        "summary": summary,
        "primary": wide,
        "all_runs_retained": True,
        "external_validation": False,
    }
    atomic_write_private_json(output / "summary.json", result)
    return result
