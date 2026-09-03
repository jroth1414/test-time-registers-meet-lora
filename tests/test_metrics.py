import math

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
