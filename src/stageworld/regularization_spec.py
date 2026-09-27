"""Full-retrain batch-size arms; no reused or imported pretrained weights."""

import os

from stageworld.surgery_s2_spec import SEEDS

BATCH_SIZES = (16, 8, 4)


def configs() -> list[dict]:
    base = {
        "world_lr": 2e-5,
        "head_surgery_lr": 2e-4,
        "alpha": 1.0,
        "surgery_blocks": 4,
        "ct_weight": 0.1,
        "weight_decay": 0.01,
        "head_dropout": 0.2,
    }
    changes = [
        ("baseline", {}),
        ("lr1_alpha05", {"world_lr": 1e-5, "head_surgery_lr": 1e-4, "alpha": 0.5}),
        ("lr1_s2b2", {"world_lr": 1e-5, "head_surgery_lr": 1e-4, "surgery_blocks": 2}),
        ("alpha05_s2b2", {"alpha": 0.5, "surgery_blocks": 2}),
        (
            "lr1_alpha05_s2b2",
            {
                "world_lr": 1e-5,
                "head_surgery_lr": 1e-4,
                "alpha": 0.5,
                "surgery_blocks": 2,
            },
        ),
    ]
    return [{"name": name, **base, **change} for name, change in changes]


def arms() -> list[dict]:
    result = []
    for batch_size in BATCH_SIZES:
        for config in configs():
            result.append(
                {
                    "name": f"bs{batch_size}_{config['name']}",
                    "batch_size": batch_size,
                    **{key: value for key, value in config.items() if key != "name"},
                }
            )
    return result


def specification(smoke: bool = False) -> dict:
    selected = os.environ.get("GENERATED651_ARM", "all")
    selected_arms = arms()
    if selected != "all":
        selected_arms = [arm for arm in selected_arms if arm["name"] == selected]
        if not selected_arms:
            raise ValueError(f"Unknown Generated651 arm: {selected}")
    return {
        "schema": "generated651-fullretrain-bs-v1",
        "smoke": smoke,
        "seeds": list(SEEDS[:1] if smoke else SEEDS),
        "arms": selected_arms,
        "batch_sizes": list(dict.fromkeys(arm["batch_size"] for arm in selected_arms)),
        "paired_baseline": "bs16_baseline" if selected == "all" else selected,
        "architecture": "g2",
        "loss": "balanced_bce",
        "sizes": {"train": 521, "validation": 65, "test": 65},
        "phase_epochs": 2 if smoke else 400,
        "patience": 50,
        "min_delta": 1e-4,
        "world_learning_rate": 2e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "threshold": 0.5,
        "world_frozen": False,
        "stage_one": "train_from_scratch_per_seed_and_batch_size",
        "training_precision": "cuda_bfloat16",
        "evaluation_precision": "float32",
        "model_selection": "max_validation_mean_endpoint_AP",
        "configuration_selection": "max_20seed_mean_validation_mean_endpoint_AP",
        "configuration_tie_break": "declared_arm_order",
        "test_gate": "all_fits_and_configuration_choice_locked_before_test_inference",
        "retention": "selected_and_final_weights_history_and_bundle; recovery_while_unfinished",
        "external_validation": False,
        "interpretation": "internal_reused_cohort_overlapping_holdouts",
    }
