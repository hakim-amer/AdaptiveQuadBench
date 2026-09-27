import torch

from icon_mpc.models.mamba import DA_MIN, MambaStack, selective_scan


def test_scan_matches_recurrence():
    torch.manual_seed(0)
    b, T, d, n = 2, 150, 5, 4
    u, dt = torch.randn(b, T, d), torch.rand(b, T, d) * 3.0
    A, B, C, D = -torch.rand(d, n) * 3, torch.randn(b, T, n), torch.randn(b, T, n), torch.randn(d)
    y, hT = selective_scan(u, dt, A, B, C, D)
    h = torch.zeros(b, d, n)
    ys = []
    for t in range(T):
        h = torch.exp((dt[:, t, :, None] * A).clamp(min=DA_MIN)) * h + dt[:, t, :, None] * B[:, t, None] * u[:, t, :, None]
        ys.append((h * C[:, t, None]).sum(-1) + D * u[:, t])
    assert torch.allclose(y, torch.stack(ys, 1), atol=1e-4)
    assert torch.allclose(hT, h, atol=1e-4)


import pytest


@pytest.mark.parametrize('selective', [True, False])
def test_step_matches_forward(selective):
    torch.manual_seed(1)
    m = MambaStack(d_in=7, d_out=3, d_model=16, n_layers=2, d_state=8, selective=selective).eval()
    x = torch.randn(3, 90, 7)
    with torch.no_grad():
        y_par = m(x)
        s = m.init_state(3)
        y_rec = []
        for t in range(90):
            yt, s = m.step(x[:, t], s)
            y_rec.append(yt)
    assert torch.allclose(y_par, torch.stack(y_rec, 1), atol=1e-4)
