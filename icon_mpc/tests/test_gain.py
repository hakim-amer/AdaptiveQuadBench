import torch

from icon_mpc.learned.features import KF_BANK, KF_START, RES_SLICE
from icon_mpc.models.gain import GainNet, gated_scan

D_IN = KF_START + 6 * len(KF_BANK)


def test_gated_scan_matches_recurrence():
    torch.manual_seed(0)
    K = torch.rand(3, 37, 6) * 0.5
    r = torch.randn(3, 37, 6)
    h = torch.zeros(3, 6)
    ref = []
    for t in range(37):
        h = (1 - K[:, t]) * h + K[:, t] * r[:, t]
        ref.append(h)
    assert torch.allclose(gated_scan(K, r, torch.zeros(3, 6)), torch.stack(ref, 1), atol=1e-5)


def test_step_matches_forward():
    torch.manual_seed(0)
    net = GainNet(D_IN, 36)
    x = torch.randn(2, 30, D_IN)
    st = net.init_state(2)
    ys = []
    for t in range(30):
        y, st = net.step(x[:, t], st)
        ys.append(y)
    assert torch.allclose(torch.stack(ys, 1), net(x), atol=1e-5)


def test_offset_free_for_random_weights():
    """Constant disturbance seen by residual and all KF experts -> output converges to it,
    whatever the (random) network weights and features."""
    torch.manual_seed(1)
    for _ in range(3):
        net = GainNet(D_IN, 36)
        net.backbone.head.weight.data.normal_(0, 2.0)  # arbitrary gain / mixture outputs
        net.backbone.head.bias.data.normal_(0, 3.0)
        x = torch.randn(1, 3000, D_IN)
        d = torch.randn(6)
        x[..., RES_SLICE] = d
        for j in range(len(KF_BANK)):
            x[..., KF_START + 6 * j:KF_START + 6 * j + 6] = d
        out = net(x)[0, -1, :6]
        assert torch.allclose(out, torch.zeros(6), atol=1e-3)  # correction to base KF == 0 -> d
