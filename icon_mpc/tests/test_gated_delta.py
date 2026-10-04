import torch

from icon_mpc.models.gated_delta import GatedDeltaStack


def test_step_matches_forward():
    torch.manual_seed(1)
    m = GatedDeltaStack(d_in=7, d_out=3, d_model=16, n_layers=2, n_heads=2, d_head=8).eval()
    x = torch.randn(3, 90, 7)
    with torch.no_grad():
        y_par = m(x)
        s = m.init_state(3)
        y_rec = []
        for t in range(90):
            yt, s = m.step(x[:, t], s)
            y_rec.append(yt)
    assert torch.allclose(y_par, torch.stack(y_rec, 1), atol=1e-4)


def test_delta_rule_solves_in_context_regression():
    """With alpha=1, beta=1 and orthonormal keys the delta rule recovers v = W k exactly."""
    from icon_mpc.models.gated_delta import GatedDeltaBlock
    torch.manual_seed(0)
    d = 8
    W = torch.randn(1, 1, d, d)
    S = torch.zeros(1, 1, d, d)
    keys = torch.eye(d)
    for i in range(d):
        k = keys[i].view(1, 1, d)
        S, _ = GatedDeltaBlock._rec(S, k, k, torch.einsum('bhvk,bhk->bhv', W, k), torch.ones(1, 1), torch.ones(1, 1))
    assert torch.allclose(S, W, atol=1e-5)


def test_chunked_matches_sequential():
    torch.manual_seed(2)
    m = GatedDeltaStack(d_in=7, d_out=3, d_model=16, n_layers=2, n_heads=2, d_head=8).double()
    for blk in m.layers:
        blk.chunk = 16
    x = torch.randn(2, 70, 7, dtype=torch.double)  # 70 = 4 chunks + padding
    h1 = h2 = m.inp(x)
    for blk in m.layers:
        h1, h2 = blk(h1), blk(h2, sequential=True)
    assert torch.allclose(h1, h2, atol=1e-9)
    # gradients agree as well
    g1 = torch.autograd.grad(m.layers[0](m.inp(x)).sum(), m.layers[0].ab.weight)[0]
    g2 = torch.autograd.grad(m.layers[0](m.inp(x), sequential=True).sum(), m.layers[0].ab.weight)[0]
    assert torch.allclose(g1, g2, atol=1e-8)
