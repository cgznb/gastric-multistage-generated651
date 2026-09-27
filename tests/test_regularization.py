import copy
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
from torch import nn

from stageworld.artifacts import atomic_write_private_json
from stageworld.generated700_data import Pool
from stageworld.generated700_models import FutureWorld, endpoint_loss
from stageworld.regularization_metrics import balanced_components
from stageworld.regularization_models import build_model
from stageworld.regularization_spec import arms, specification
from stageworld.regularization_training import train_fit
from stageworld.regularization_workflow import evaluate_tests, freeze_selection, write_results
from stageworld.surgery_s2_data import SurgeryPool
from stageworld.surgery_s2_evaluation import endpoint_metrics
from stageworld.surgery_s2_io import object_hash
from stageworld.surgery_s2_models import SurgeryClassifier
from stageworld.synthetic_workflow import _atomic_torch_save


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)


def test_declared_matrix_and_baseline_parity():
    configs = arms()
    assert len(configs) == 15 and len(specification()["seeds"]) == 20
    assert {c["batch_size"] for c in configs} == {16, 8, 4}
    assert len({c["name"] for c in configs}) == 15
    bs16 = [c for c in configs if c["batch_size"] == 16]
    assert [c["name"] for c in bs16] == [
        "bs16_baseline",
        "bs16_lr1_alpha05",
        "bs16_lr1_s2b2",
        "bs16_alpha05_s2b2",
        "bs16_lr1_alpha05_s2b2",
    ]
    anchor = {"weight": torch.randn(2, 361), "bias": torch.randn(2)}
    torch.manual_seed(17)
    original = SurgeryClassifier(361, "g2", anchor["weight"], anchor["bias"], 8, 17)
    model = build_model(bs16[0], 8, 17, anchor)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.state_dict().items())
    shallow = build_model(next(c for c in configs if c["name"] == "bs16_alpha05_s2b2"), 8, 17, anchor)
    assert len(shallow.surgery.blocks) == 2
    assert torch.equal(shallow.residual_scale, torch.full((2,), 0.5))
    assert all(
        torch.equal(v, model.state_dict()[k])
        for k, v in shallow.state_dict().items()
        if k != "residual_scale"
    )
    dropped = build_model(next(c for c in configs if c["name"] == "bs16_lr1_alpha05_s2b2"), 8, 17, anchor)
    assert all(
        m.p == 0.2 for head in dropped.endpoints for m in head.body if isinstance(m, nn.Dropout)
    )
    assert dropped.surgery.blocks[0].transformer.dropout.p == 0.1


def test_residual_scale_and_prefix_gradients():
    configs = arms()
    anchor = {"weight": torch.randn(2, 361) * 0.01, "bias": torch.randn(2) * 0.01}
    full = build_model(configs[0], 8, 17, anchor).eval()
    half = build_model(next(c for c in configs if c["name"] == "bs16_lr1_alpha05"), 8, 17, anchor).eval()
    for model in (full, half):
        for head in model.endpoints:
            nn.init.constant_(head.body[-1].weight, 0.02)
        nn.init.constant_(model.surgery.delta[-1].weight, 0.01)
    x, ct, valid, status = (
        torch.randn(4, 361),
        torch.randn(4, 27, 8),
        torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.long),
    )
    a, b = full.forward_details(x, ct, valid, status), half.forward_details(x, ct, valid, status)
    z = torch.nn.functional.linear(x, anchor["weight"], anchor["bias"])
    assert torch.allclose(b["logits"] - z, 0.5 * (a["logits"] - z), atol=1e-7)
    b["logits"][:, 0].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in half.surgery.parameters())
    half.zero_grad(set_to_none=True)
    half(x, ct, valid, status)[:, 1].sum().backward()
    assert any(p.grad is not None and p.grad.any() for p in half.surgery.parameters())
    assert any(p.grad is not None and p.grad.any() for p in half.world.parameters())
    assert not half.anchor_weight.requires_grad and not half.residual_scale.requires_grad


def test_component_logging_matches_loss_and_gradient():
    logits = torch.randn(9, 2, requires_grad=True)
    labels, valid = torch.randint(0, 2, (9, 2)).float(), torch.ones(9, 2, dtype=torch.bool)
    weights = torch.tensor([3.0, 4.0])
    combined = endpoint_loss(logits, labels, valid, weights)
    components = sum(balanced_components(logits, labels, valid, weights))
    assert torch.allclose(combined, components)
    assert torch.equal(
        torch.autograd.grad(combined, logits, retain_graph=True)[0],
        torch.autograd.grad(components, logits)[0],
    )


def synthetic_context():
    ids = [f"synthetic-{i}" for i in range(12)]
    ct0, ct1 = torch.randn(12, 27, 8), torch.randn(12, 27, 8)
    labels = torch.tensor([[i % 2, (i // 2) % 2] for i in range(12)]).float()
    base = Pool(
        ids,
        {},
        {},
        torch.full((12,), 90.0),
        ct0,
        torch.ones(12, dtype=torch.bool),
        ct1.mean(1),
        torch.ones(12, dtype=torch.bool),
        labels,
        torch.ones(12, 2, dtype=torch.bool),
        "synthetic-pool",
        ct1,
    )
    pool = SurgeryPool(base, torch.ones(12, 3, dtype=torch.long), "synthetic-events")
    snapshot = {"artifact_id": "synthetic-input", "fit_ids": ids[:8]}
    anchor = {
        "artifact_id": "synthetic-anchor",
        "contract": {"snapshot_id": snapshot["artifact_id"]},
        "weight": torch.zeros(2, 361),
        "bias": torch.zeros(2),
    }
    spec = specification(True)
    spec.update({"phase_epochs": 3})
    spec["arms"] = [{**config, "batch_size": 4} for config in spec["arms"]]
    world = FutureWorld(361, 8)
    world.fit_statistics(ct0[:8], ct1[:8])
    parent = {
        "artifact_id": "synthetic-parent-model",
        "model_state": world.state_dict(),
        "contract": {
            "seed": 17,
            "architecture": "world",
            "snapshot_id": snapshot["artifact_id"],
            "train_ids": ids[:8],
            "validation_ids": ids[8:],
            "protocol_id": object_hash(spec),
        },
    }
    return pool, torch.randn(12, 361), snapshot, anchor, spec, parent


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_partial_epoch_recovery_and_retention(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pool, x, snapshot, anchor, spec, parent = synthetic_context()
    args = (pool, x, snapshot, torch.arange(8), torch.arange(8, 12))
    kwargs = {
        "seed": 17,
        "config": spec["arms"][0],
        "spec": spec,
        "anchor": anchor,
        "parent": parent,
        "device": device,
    }
    a, _ = train_fit(*args, tmp_path / "a", **kwargs)
    with pytest.raises(RuntimeError, match="intentional_partial"):
        train_fit(*args, tmp_path / "b", interrupt_after_update=3, **kwargs)
    assert (tmp_path / "b/recovery.pt").exists()
    b, _ = train_fit(*args, tmp_path / "b", **kwargs)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    ca = torch.load(tmp_path / "a/final.pt", weights_only=True)
    cb = torch.load(tmp_path / "b/final.pt", weights_only=True)
    assert ca["history"] == cb["history"]
    assert all(torch.equal(v, cb["model_state"][k]) for k, v in ca["model_state"].items())
    assert "optimizer_state" not in ca and not (tmp_path / "b/recovery.pt").exists()
    before = (tmp_path / "b/final.pt").stat().st_mtime_ns
    train_fit(*args, tmp_path / "b", **kwargs)
    assert before == (tmp_path / "b/final.pt").stat().st_mtime_ns


def test_partition_parent_and_test_gates(tmp_path):
    pool, x, snapshot, anchor, spec, parent = synthetic_context()
    kwargs = {
        "seed": 17,
        "config": spec["arms"][0],
        "spec": spec,
        "anchor": anchor,
        "parent": parent,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="isolated"):
        train_fit(
            pool, x, snapshot, torch.arange(8), torch.arange(8, 10), tmp_path / "hidden", **kwargs
        )
    broken = copy.deepcopy(parent)
    broken["contract"]["seed"] = 503
    with pytest.raises(ValueError, match="parent"):
        train_fit(
            pool,
            x,
            snapshot,
            torch.arange(8),
            torch.arange(8, 12),
            tmp_path / "parent",
            **{**kwargs, "parent": broken},
        )
    with pytest.raises(ValueError, match="world pretrains"):
        freeze_selection(tmp_path, spec)
    with pytest.raises(ValueError, match="Smoke"):
        evaluate_tests(pool, tmp_path, spec)
    with pytest.raises(ValueError, match="coverage"):
        write_results(tmp_path, specification(), [], [], [])


def test_global_validation_choice_and_lock(tmp_path, monkeypatch):
    import stageworld.regularization_workflow as workflow

    spec = specification()
    manifest = {"entries": []}
    spec.update({"seeds": [17, 43], "imports_id": object_hash(manifest), "source_sha256": "fixed"})
    atomic_write_private_json(tmp_path / "imports.json", manifest)
    monkeypatch.setattr(workflow, "source_hash", lambda _: "fixed")
    for seed in spec["seeds"]:
        for batch_size in spec["batch_sizes"]:
            world = tmp_path / "world" / f"bs{batch_size}" / f"seed-{seed}"
            world.mkdir(parents=True)
            _atomic_torch_save(world / "selected.pt", {"synthetic": True})
            atomic_write_private_json(world / "completed.json", {"status": "completed"})
        for index, config in enumerate(spec["arms"]):
            phase = tmp_path / "fits" / config["name"] / f"seed-{seed}"
            score = 0.4 if index == 0 else 0.5 if index in (1, 2) else 0.3
            _atomic_torch_save(
                phase / "selected.pt",
                {
                    "artifact_id": "synthetic-selected",
                    "contract": {"protocol_id": object_hash(spec), "arm": config, "seed": seed},
                },
            )
            for name in ("inference.pt", "validation.pt"):
                _atomic_torch_save(phase / name, {"synthetic": True})
            atomic_write_private_json(phase / "history.json", {"epochs": []})
            atomic_write_private_json(
                phase / "completed.json",
                {
                    "status": "completed",
                    "selected_id": "synthetic-selected",
                    "validation": {"mean_AP": score},
                },
            )
    locked = freeze_selection(tmp_path, spec)
    assert locked["choice"]["selected_arm"] == spec["arms"][1]["name"]
    assert locked["choice"]["test_used"] is False
    assert not (tmp_path / "predictions").exists()
    assert freeze_selection(tmp_path, spec) == locked


def test_complete_report_and_undefined_metrics(tmp_path):
    import csv

    spec = specification()
    atomic_write_private_json(tmp_path / "configuration_choice.json", {"selected_arm": "baseline"})
    labels = torch.tensor([[i % 5 == 0, i % 4 == 0] for i in range(65)]).float()
    metrics = endpoint_metrics(torch.full((65, 2), 0.2, dtype=torch.float64), labels)
    rows = [
        {"seed": seed, "arm": arm, "endpoint": endpoint, "partition": "test", **value}
        for seed in spec["seeds"]
        for arm in [c["name"] for c in spec["arms"]] + ["logistic"]
        for endpoint, value in metrics.items()
    ]
    report = write_results(tmp_path, spec, rows, [{"synthetic": True}], [])
    assert report["test_endpoint_rows"] == 640
    with (tmp_path / "evaluation/all_configs_per_seed.csv").open() as handle:
        wide = list(csv.DictReader(handle))
    assert len(wide) == 320 and all(r["pcr_precision"] == "" for r in wide)
    with (tmp_path / "evaluation/paired_vs_baseline.csv").open() as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 560 and all(float(r["delta_auprc"]) == 0 for r in paired)
