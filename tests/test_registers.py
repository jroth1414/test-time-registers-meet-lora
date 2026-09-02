import math
from pathlib import Path

import torch

from ttr.backbone import tiny_backbone
from ttr.registers import (
    OutlierStats,
    RegisterNeurons,
    attention_entropy,
    calibrate_outlier_threshold,
    find_register_neurons,
    install_test_time_registers,
    load_register_neurons,
    outlier_fraction,
    outlier_mask,
    patch_norms,
    save_register_neurons,
    score_register_neurons,
    select_register_neurons,
)


def _loader(n_batches=3, b=2, img=56, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, 3, img, img, generator=g) for _ in range(n_batches)]


def test_patch_norms_shape():
    bb = tiny_backbone(depth=2)
    n = patch_norms(bb, torch.randn(2, 3, 56, 56), layer=-1)
    assert n.shape == (2, 16)
    assert (n > 0).all()


def test_calibrate_returns_threshold_above_median():
    bb = tiny_backbone(depth=2)
    st = calibrate_outlier_threshold(bb, _loader(), layer=-1, k=4.0, max_images=4)
    assert isinstance(st, OutlierStats)
    assert st.layer == 1  # -1 resolved to the last block
    assert st.tau > st.median
    assert st.mad >= 0


def test_random_model_has_near_zero_outliers_and_planted_outlier_is_found():
    bb = tiny_backbone(depth=2)
    st = calibrate_outlier_threshold(bb, _loader(), max_images=6)
    frac = outlier_fraction(bb, _loader(seed=1), st)
    assert 0.0 <= frac < 0.1

    # Plant an artifact: scale patch token 5 by 50x at the output of the last block.
    def boost(module, inp, out):
        out = out.clone()
        out[:, bb.prefix_len() + 5] *= 50.0
        return out

    h = bb.model.blocks[-1].register_forward_hook(boost)
    m = outlier_mask(bb, torch.randn(2, 3, 56, 56), st)
    h.remove()
    assert m.shape == (2, 16)
    assert m[:, 5].all()
    assert m.sum() <= 4  # token 5 in both images, maybe a stray one or two


def test_score_is_high_for_neuron_active_only_on_outliers():
    # acts[layer]: (B, P, H) patch-token activations; outlier: (B, P) bool
    B, P, H = 2, 16, 8
    g = torch.Generator().manual_seed(0)
    acts = {0: torch.randn(B, P, H, generator=g) * 0.1}
    outlier = torch.zeros(B, P, dtype=torch.bool)
    outlier[0, 3] = True
    outlier[1, 7] = True
    acts[0][0, 3, 5] = 10.0  # neuron 5 fires only on the outlier tokens
    acts[0][1, 7, 5] = 12.0
    scores = score_register_neurons(acts, outlier)
    assert scores[0].shape == (H,)
    assert scores[0].argmax().item() == 5
    assert scores[0][5] > 5 * scores[0].topk(2).values[1]


def test_select_respects_quantile_and_cap():
    scores = {0: torch.tensor([0.0, 1.0, 9.0, 0.5]), 1: torch.tensor([8.0, 0.0, 0.0, 7.0])}
    sel = select_register_neurons(scores, quantile=0.5, max_neurons=2)
    assert sel == {0: [2], 1: [0]}  # top-2 overall, sorted per layer
    sel_all = select_register_neurons(scores, quantile=0.0, max_neurons=100)
    assert sel_all == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}


def test_score_with_no_outliers_returns_zeros():
    acts = {0: torch.randn(1, 4, 3)}
    scores = score_register_neurons(acts, torch.zeros(1, 4, dtype=torch.bool))
    assert torch.all(scores[0] == 0)


def test_find_register_neurons_end_to_end_with_planted_neuron(tmp_path: Path):
    bb = tiny_backbone(depth=2, embed_dim=32, mlp_ratio=2.0)  # hidden = 64
    loader = _loader(n_batches=4, b=2, seed=3)
    st = calibrate_outlier_threshold(bb, loader, max_images=8)

    # Plant: make patch token 2 an outlier at the last block AND make neuron 9 of layer 0
    # fire hard on that token. The detector must recover layer 0 / neuron 9.
    def boost_resid(module, inp, out):
        out = out.clone()
        out[:, bb.prefix_len() + 2] *= 50.0
        return out

    def fire_neuron(module, inp, out):
        out = out.clone()
        out[:, bb.prefix_len() + 2, 9] += 20.0
        return out

    h1 = bb.model.blocks[-1].register_forward_hook(boost_resid)
    h2 = bb.add_mlp_hook(0, fire_neuron)
    rn = find_register_neurons(bb, loader, st, quantile=0.99, max_neurons=4, max_images=8)
    h1.remove()
    h2.remove()

    assert isinstance(rn, RegisterNeurons)
    assert 9 in rn.layer_to_neurons.get(0, [])

    p = tmp_path / "rn.json"
    save_register_neurons(rn, p)
    back = load_register_neurons(p)
    assert back.layer_to_neurons == rn.layer_to_neurons
    assert back.stats == rn.stats


def test_find_register_neurons_returns_empty_when_no_outliers():
    bb = tiny_backbone(depth=2)
    loader = _loader(n_batches=2, b=2, seed=5)
    st = OutlierStats(layer=1, tau=1e9, median=0.0, mad=0.0, k=4.0)  # nothing exceeds tau
    rn = find_register_neurons(bb, loader, st, quantile=0.0, max_neurons=64, max_images=4)
    assert rn.layer_to_neurons == {}


def test_install_moves_neuron_activation_to_register_token():
    bb = tiny_backbone(depth=2, embed_dim=32, mlp_ratio=2.0)
    bb.set_tt_registers(1)
    st = OutlierStats(layer=1, tau=1e9, median=0.0, mad=0.0, k=4.0)
    rn = RegisterNeurons(layer_to_neurons={0: [1, 2]}, stats=st)
    handles = install_test_time_registers(bb, rn)
    cap = bb.capture("mlp_act", layers=[0])  # registered AFTER the intervention
    x = torch.randn(2, 3, 56, 56)
    bb.forward_tokens(x)
    a = cap.data[0]
    ps, rs = bb.patch_slice(), bb.tt_reg_slice()
    assert torch.all(a[:, ps][..., [1, 2]] == 0)  # zeroed on patches
    # register token now carries the max of what the patches had
    for h in handles:
        h.remove()
    cap.remove()
    cap2 = bb.capture("mlp_act", layers=[0])
    bb.forward_tokens(x)
    raw = cap2.data[0]
    cap2.remove()
    expected = raw[:, ps][..., [1, 2]].max(dim=1).values
    assert torch.allclose(a[:, rs][:, 0, [1, 2]], expected)


def test_install_requires_tt_registers():
    import pytest

    bb = tiny_backbone()
    rn = RegisterNeurons({0: [0]}, OutlierStats(0, 1.0, 0.0, 0.0, 4.0))
    with pytest.raises(RuntimeError):
        install_test_time_registers(bb, rn)


def test_install_with_two_registers_round_robins_neurons():
    bb = tiny_backbone(depth=2, mlp_ratio=2.0)
    bb.set_tt_registers(2)
    x = torch.randn(1, 3, 56, 56)
    # Un-intervened activations at layer 0 (the register tokens are not zero here: they pick
    # up the attention residual before the MLP).
    raw_cap = bb.capture("mlp_act", layers=[0])
    bb.forward_tokens(x)
    raw = raw_cap.data[0]
    raw_cap.remove()
    ps, rs = bb.patch_slice(), bb.tt_reg_slice()
    peaks = raw[:, ps].max(dim=1).values  # (1, hidden)

    rn = RegisterNeurons({0: [4, 5, 6]}, OutlierStats(1, 1e9, 0.0, 0.0, 4.0))
    handles = install_test_time_registers(bb, rn)
    cap = bb.capture("mlp_act", layers=[0])
    bb.forward_tokens(x)
    a = cap.data[0][:, rs]  # (1, 2, hidden)
    # neuron 4 -> register 0, neuron 5 -> register 1, neuron 6 -> register 0 carry the peaks
    assert a[0, 0, 4] == peaks[0, 4] and a[0, 1, 5] == peaks[0, 5] and a[0, 0, 6] == peaks[0, 6]
    # the other register/neuron pairs are untouched
    assert a[0, 1, 4] == raw[0, rs.start + 1, 4]
    assert a[0, 0, 5] == raw[0, rs.start + 0, 5]
    assert a[0, 1, 6] == raw[0, rs.start + 1, 6]
    for h in handles:
        h.remove()
    cap.remove()


def test_attention_entropy_keys_and_range():
    bb = tiny_backbone(depth=2, heads=2)
    bb.set_tt_registers(1)
    ent = attention_entropy(bb, _loader(n_batches=2), layers=[1], max_images=4)
    T = 1 + 1 + 16
    for key in ("cls", "tt_reg", "patch"):
        assert 0.0 <= ent[key] <= math.log(T) + 1e-6
    assert bb.model.blocks[1].attn.fused_attn is False  # left disabled on purpose
