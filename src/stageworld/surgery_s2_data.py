"""Verified event join and physically isolated development partitions."""

from dataclasses import dataclass, replace

import torch

from stageworld.generated700_data import Pool, load_pool, raw_tabular
from stageworld.surgery_s2_spec import EVENT_ROOT, PRESENT, SOURCE_ROOT


@dataclass
class SurgeryPool:
    base: Pool
    events: torch.Tensor
    artifact_id: str

    @property
    def surgery(self) -> torch.Tensor:
        return self.events[:, 1]


def load_verified_pool() -> SurgeryPool:
    base = load_pool(SOURCE_ROOT / "artifacts/pool")
    event = torch.load(
        EVENT_ROOT / "artifacts/pool/events.pt", weights_only=True, map_location="cpu"
    )
    ids = event["patient_ids"]
    if (
        event["source_pool_id"] != base.artifact_id
        or len(set(ids)) != len(ids)
        or set(ids) != set(base.ids)
        or len(ids) != 651
    ):
        raise ValueError("Surgery cache and original complete651 cohort do not match")
    lookup = {patient: i for i, patient in enumerate(ids)}
    events = event["events"][[lookup[patient] for patient in base.ids]].clone()
    if (
        events.shape != (651, 3)
        or events.dtype != torch.long
        or not ((events >= 0) & (events < 4)).all()
        or not (events[:, 1] == PRESENT).all()
        or not base.ct0_valid.all()
        or not base.ct1_valid.all()
        or not torch.isfinite(base.ct1_tokens).all()
    ):
        raise ValueError("Complete651 image/event audit failed")
    return SurgeryPool(base, events, event["artifact_id"])


def subset_pool(pool: SurgeryPool, ids: list[str]) -> SurgeryPool:
    if len(ids) != len(set(ids)) or not set(ids) <= set(pool.base.ids):
        raise ValueError("A partition requires unique known patient identifiers")
    rows = pool.base.indices(ids)
    fields = {
        name: getattr(pool.base, name)[rows].clone()
        for name in (
            "interval",
            "ct0",
            "ct0_valid",
            "ct1",
            "ct1_valid",
            "labels",
            "valid",
            "ct1_tokens",
        )
    }
    base = replace(
        pool.base,
        ids=list(ids),
        clinical={p: pool.base.clinical[p] for p in ids},
        treatments={p: pool.base.treatments[p] for p in ids},
        **fields,
    )
    return SurgeryPool(base, pool.events[rows].clone(), pool.artifact_id)


def encode_inputs(pool: Pool, snapshot: dict) -> torch.Tensor:
    raw = torch.from_numpy(raw_tabular(pool, snapshot["clinical"], snapshot["support"]))
    x = ((raw - snapshot["mean"]) / snapshot["scale"]).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite preprocessed input")
    return x
