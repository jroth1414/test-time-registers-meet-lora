import torch

from ttr.backbone import tiny_backbone
from ttr.registers import (
    OutlierStats,
    calibrate_outlier_threshold,
    outlier_fraction,
    outlier_mask,
    patch_norms,
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
