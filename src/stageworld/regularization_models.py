"""Small parameterized adapter preserving the original baseline initialization."""

import torch
from torch import nn

from stageworld.generated_workflow import _seed
from stageworld.surgery_s2_models import SurgeryClassifier


def build_model(config, image_dim, seed, anchor, parent=None):
    if config["surgery_blocks"] not in (1, 2, 4) or config["alpha"] not in (0.5, 1.0):
        raise ValueError("Unregistered depth or residual coefficient")
    if config["head_dropout"] not in (0.2, 0.3):
        raise ValueError("Unregistered head dropout")
    torch.backends.mha.set_fastpath_enabled(False)
    _seed(seed)
    model = SurgeryClassifier(
        361, "g2", anchor["weight"], anchor["bias"], image_dim=image_dim, seed=seed
    )
    model.surgery.blocks = nn.ModuleList(list(model.surgery.blocks)[: config["surgery_blocks"]])
    model.residual_scale.fill_(config["alpha"])
    for head in model.endpoints:
        for layer in head.body:
            if isinstance(layer, nn.Dropout):
                layer.p = config["head_dropout"]
    if parent is not None:
        model.world.load_state_dict(parent["model_state"], strict=True)
    return model
