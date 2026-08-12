from types import SimpleNamespace

import torch

from model.grasp_arch import RelationGraphProjector


def test_relation_projector_masks_padding_and_handles_empty_context():
    torch.manual_seed(0)
    config = SimpleNamespace(mm_hidden_size=6, hidden_size=8, relation_graph_tokens=4)
    projector = RelationGraphProjector(config).eval()
    values = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    first = projector(values, mask)
    changed = values.clone()
    changed[0, 1:] = 10_000
    second = projector(changed, mask)
    assert first.shape == (2, 4, 8)
    assert torch.allclose(first[0], second[0], atol=1e-6)
    assert torch.allclose(first[1], projector.null_tokens, atol=1e-6)
    assert torch.isfinite(first).all()

