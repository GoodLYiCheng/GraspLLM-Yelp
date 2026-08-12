import torch
import torch.nn as nn

from gnn.multi_gpu_inference import _sage_mean_chunked, parse_gpu_ids


class _FakeSAGE(nn.Module):
    """Minimal default-SAGE surface used by the memory-bounded kernel."""

    def __init__(self):
        super().__init__()
        self.project = False
        self.aggr = "mean"
        self.root_weight = True
        self.normalize = False
        self.out_channels = 3
        self.lin_l = nn.Linear(5, 3)
        self.lin_r = nn.Linear(5, 3, bias=False)


def test_chunked_sage_matches_pyg_reference_on_cpu():
    torch.manual_seed(7)
    conv = _FakeSAGE().eval()
    x = torch.randn(7, 5)
    # Deliberately unsorted targets: the production path sorts them first.
    edge = torch.tensor([[0, 2, 4, 1, 3, 6, 5], [2, 1, 2, 5, 5, 0, 5]])
    order = torch.argsort(edge[1], stable=True)
    actual = _sage_mean_chunked(conv, x, edge[:, order], device=torch.device("cpu"), chunk_size=2)
    aggregate = torch.zeros_like(x)
    aggregate.index_add_(0, edge[1], x[edge[0]])
    degree = torch.bincount(edge[1], minlength=x.size(0)).to(x.dtype).clamp_min_(1)
    expected = conv.lin_l(aggregate / degree.unsqueeze(1)) + conv.lin_r(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_parse_gpu_ids_rejects_duplicate_before_cuda_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert parse_gpu_ids("0, 2") == [0, 2]
    try:
        parse_gpu_ids("0,0")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate logical GPU IDs must fail")
