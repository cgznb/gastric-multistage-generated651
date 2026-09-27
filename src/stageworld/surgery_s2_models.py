"""Original Generated651 plus one surgery-conditioned latent transition."""

import torch
from torch import nn
from torch.nn import functional as F

from stageworld.generated700_models import AnchoredClassifier, SpatialTransition
from stageworld.surgery_s2_spec import CONFLICT, PRESENT, UNKNOWN


class SurgeryTransition(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.clinical = nn.Linear(32, hidden)
        self.event = nn.Embedding(4, hidden)
        self.types = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
        self.blocks = nn.ModuleList([SpatialTransition(hidden, 2) for _ in range(4)])
        self.delta = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self, s1: torch.Tensor, clinical: torch.Tensor, status: torch.Tensor
    ) -> torch.Tensor:
        if (
            status.shape != (len(s1),)
            or status.dtype != torch.long
            or not ((status >= 0) & (status < 4)).all()
        ):
            raise ValueError("Surgery status must be absent/present/unknown/conflict")
        present = status == PRESENT
        if not present.any():
            return s1
        conditions = (
            torch.stack((self.clinical(clinical[present]), self.event(status[present])), 1)
            + self.types
        )
        tokens = torch.cat((conditions, s1[present]), 1)
        for block in self.blocks:
            tokens = block(tokens)
        # No future observations or invented second time interval enter this transition.
        result = s1.clone()
        result[present] = s1[present] + self.delta(tokens[:, 2:])
        return result


class SurgeryClassifier(AnchoredClassifier):
    def __init__(
        self,
        width: int,
        architecture: str,
        anchor_weight: torch.Tensor,
        anchor_bias: torch.Tensor,
        image_dim: int = 768,
        seed: int = 17,
    ):
        if architecture not in ("g1", "g2"):
            raise ValueError("Unknown Generated651 comparison arm")
        super().__init__(width, "generated", anchor_weight, anchor_bias, image_dim)
        self.architecture = architecture
        self.surgery = None
        if architecture == "g2":
            # Preserve common initialization and the RNG stream when adding the new branch.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 424242)
                self.surgery = SurgeryTransition(self.world.hidden)

    def forward_details(
        self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor, status: torch.Tensor
    ) -> dict:
        ct0 = torch.where(ct0_valid[:, None, None], ct0, torch.zeros_like(ct0))
        initial, s1, features = self.world(x, ct0)
        pcr = self.endpoints[0](x, initial, s1)
        s2 = s1 if self.surgery is None else self.surgery(s1, x[:, :32], status)
        recurrence = self.endpoints[1](x, initial, s2)
        delta = torch.stack((pcr, recurrence), 1).float()
        anchor = F.linear(x.float(), self.anchor_weight, self.anchor_bias)
        logits = anchor + delta * ct0_valid[:, None] * self.residual_scale
        return {
            "logits": logits,
            "features": features,
            "s1": s1,
            "s2": s2,
            "delta": delta,
            "history_incomplete": (status == UNKNOWN) | (status == CONFLICT),
        }

    def forward_with_features(
        self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor, status: torch.Tensor
    ):
        result = self.forward_details(x, ct0, ct0_valid, status)
        return result["logits"], result["features"]

    def forward(
        self, x: torch.Tensor, ct0: torch.Tensor, ct0_valid: torch.Tensor, status: torch.Tensor
    ):
        return self.forward_with_features(x, ct0, ct0_valid, status)[0]
