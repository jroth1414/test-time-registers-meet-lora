import math

import pytest
import torch

from ttr.metrics import ConfusionMeter, background_fraction, image_miou


def test_confusion_meter_hand_example():
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[0, 0, 1, 1], [2, 2, 255, 255]]])
    pred = torch.tensor([[[0, 1, 1, 1], [2, 0, 0, 0]]])
    m.update(pred, target)
    # class0: tp=1, fp=1(pred0,tgt2), fn=1(pred1,tgt0) -> 1/3
    # class1: tp=2, fp=1, fn=0 -> 2/3 ; class2: tp=1, fp=0, fn=1 -> 1/2
    ious = m.per_class_iou()
    assert (
        math.isclose(ious[0], 1 / 3) and math.isclose(ious[1], 2 / 3) and math.isclose(ious[2], 0.5)
    )
    assert math.isclose(m.miou(), (1 / 3 + 2 / 3 + 0.5) / 3)
    assert math.isclose(m.pixel_acc(), 4 / 6)


def test_confusion_meter_skips_absent_classes_and_resets():
    m = ConfusionMeter(num_classes=4)
    m.update(torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]))
    assert m.miou() == 1.0
    assert math.isnan(m.per_class_iou()[3])
    m.reset()
    assert math.isnan(m.miou())


def test_image_miou_matches_meter():
    pred = torch.tensor([[0, 1], [1, 1]])
    target = torch.tensor([[0, 0], [1, 255]])
    assert math.isclose(image_miou(pred, target, 2), (0.5 + 0.5) / 2)


def test_background_fraction():
    target = torch.tensor([[0, 0, 5], [255, 2, 2]])
    assert math.isclose(background_fraction(target, [0, 2]), 4 / 5)
    assert math.isnan(background_fraction(torch.full((2, 2), 255), [0]))


def test_measure_throughput_positive():
    from ttr.metrics import measure_throughput

    ips = measure_throughput(lambda x: x * 2, torch.zeros(4, 3), iters=3, warmup=1)
    assert ips > 0


def test_confusion_meter_label_range_guard_prediction():
    """Prediction >= num_classes raises ValueError."""
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[0, 1]]])
    pred = torch.tensor([[[0, 7]]])
    with pytest.raises(ValueError, match="out of range"):
        m.update(pred, target)


def test_confusion_meter_label_range_guard_target():
    """Target >= num_classes raises ValueError."""
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[0, 5]]])
    pred = torch.tensor([[[0, 1]]])
    with pytest.raises(ValueError, match="out of range"):
        m.update(pred, target)


def test_confusion_meter_label_range_guard_all_ignore():
    """All-ignore batch does not raise and leaves matrix zero."""
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[255, 255]]])
    pred = torch.tensor([[[0, 1]]])
    m.update(pred, target)
    assert (m.mat == 0).all()


def test_confusion_meter_accumulation():
    """Two updates on disjoint halves equal one update on whole."""
    m1 = ConfusionMeter(num_classes=3)
    m2 = ConfusionMeter(num_classes=3)

    full_target = torch.tensor([[[0, 1, 2, 0, 1, 2]]])
    full_pred = torch.tensor([[[0, 1, 2, 1, 0, 1]]])

    # First half
    m1.update(full_pred[:, :, :3], full_target[:, :, :3])
    # Second half
    m1.update(full_pred[:, :, 3:], full_target[:, :, 3:])

    # All at once
    m2.update(full_pred, full_target)

    assert torch.equal(m1.mat, m2.mat)


def test_image_miou_all_ignore():
    """image_miou on all-255 target returns NaN."""
    pred = torch.tensor([[0, 1], [1, 1]])
    target = torch.full((2, 2), 255)
    result = image_miou(pred, target, 2)
    assert math.isnan(result)


def test_confusion_meter_pixel_acc_empty():
    """pixel_acc() on fresh meter returns NaN."""
    m = ConfusionMeter(num_classes=3)
    result = m.pixel_acc()
    assert math.isnan(result)


def test_confusion_meter_per_class_iou_edge_cases():
    """per_class_iou edge cases: absent from both is NaN, predicted but not in GT is 0.0."""
    m = ConfusionMeter(num_classes=3)
    target = torch.tensor([[[0, 0, 1, 1]]])
    pred = torch.tensor([[[0, 0, 1, 1]]])
    m.update(pred, target)

    ious = m.per_class_iou()
    # Class 0: tp=2, union=2 -> 1.0
    # Class 1: tp=2, union=2 -> 1.0
    # Class 2: tp=0, union=0 -> NaN (absent from both)
    assert math.isclose(ious[0], 1.0)
    assert math.isclose(ious[1], 1.0)
    assert math.isnan(ious[2])
