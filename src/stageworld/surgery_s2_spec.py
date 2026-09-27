"""Prespecified twenty-seed patient holdout experiment, superseding fivefold CV."""

import os
from pathlib import Path

SEEDS = (
    17,
    43,
    97,
    131,
    173,
    211,
    257,
    307,
    359,
    419,
    461,
    503,
    547,
    601,
    653,
    701,
    751,
    809,
    853,
    907,
)
ARCHITECTURES = ("g1", "g2")
LOSSES = ("balanced", "bce", "focal")
ABSENT, PRESENT, UNKNOWN, CONFLICT = range(4)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("GENERATED651_SOURCE_ROOT", PROJECT_ROOT / "data/complete651"))
EVENT_ROOT = Path(os.environ.get("GENERATED651_EVENT_ROOT", PROJECT_ROOT / "data/surgery"))


def specification(smoke: bool = False) -> dict:
    return {
        "schema": "generated651-surgery-s2-holdout-v1",
        "smoke": smoke,
        "seeds": list(SEEDS[:1] if smoke else SEEDS),
        "architectures": list(ARCHITECTURES),
        "losses": list(LOSSES),
        "primary_comparison": "g2_balanced_minus_g1_balanced",
        "patients": 651,
        "split": "per_seed_joint_stratified_80_10_10",
        "sizes": {"train": 521, "validation": 65, "test": 65},
        "phase_epochs": 2 if smoke else 100,
        "patience": 15,
        "min_delta": 0.0001,
        "batch_size": 32,
        "world_learning_rate": 0.0002,
        "joint_world_learning_rate": 0.00002,
        "joint_head_surgery_learning_rate": 0.0002,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "training_precision": "cuda_bfloat16",
        "evaluation_precision": "float32",
        "ct_loss_coefficient": 0.1,
        "world_selection": "min_validation_ct_feature_set_loss",
        "joint_selection": "max_validation_mean_endpoint_AP",
        "threshold": 0.5,
        "residual_scale": 1.0,
        "anchor_C": 1.0,
        "anchor_class_weight": None,
        "surgery_conditions": ["baseline_clinical", "surgery_status"],
        "surgery_blocks": 4,
        "surgery_output_initialization": "zero_residual",
        "surgery_initialization_seed_offset": 424242,
        "batch_order": "independent_cpu_generator_seed_times_100003_plus_epoch",
        "test_gate": "all_seeds_all_arms_selection_locked_before_test_scoring",
        "endpoint": "recorded_recurrence_or_metastasis_status",
        "all_patients_surgery_present": True,
        "external_validation": False,
        "interpretation": "internal_reused_cohort_holdout_not_independent_across_seeds",
    }
