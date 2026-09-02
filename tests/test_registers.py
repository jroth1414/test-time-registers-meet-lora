import torch

from ttr.backbone import tiny_backbone
from ttr.registers import (
    OutlierStats,
    calibrate_outlier_threshold,
    outlier_fraction,
    outlier_mask,
    patch_norms,
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
