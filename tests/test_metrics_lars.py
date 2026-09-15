import math

import torch

from ttr.metrics_lars import boundary_f1, lars_extra_metrics, pixel_f1


def _scene(edge_row):
    t = torch.full((32, 32), 1)  # water
    t[:edge_row] = 2  # sky above the edge
    return t


def test_boundary_f1_perfect_and_shifted():
    t = _scene(10)
    assert math.isclose(boundary_f1(t, t, class_id=1, tol=2), 1.0)
    assert boundary_f1(_scene(11), t, class_id=1, tol=2) == 1.0  # within tolerance
    assert boundary_f1(_scene(16), t, class_id=1, tol=2) == 0.0  # outside tolerance


def test_pixel_f1_obstacle():
    t = torch.full((8, 8), 1)
    t[2:4, 2:4] = 0
    p = t.clone()
    p[2:4, 2:6] = 0  # 4 tp, 4 fp, 0 fn -> precision .5 recall 1 -> f1 .667
    assert math.isclose(pixel_f1(p, t, 0), 2 / 3)
    assert math.isnan(pixel_f1(torch.ones(4, 4), torch.ones(4, 4), 0))


def test_lars_extra_metrics_keys():
    t = _scene(10)
    out = lars_extra_metrics(t, t)
    assert set(out) == {"water_edge_f1", "obstacle_f1"}


def _scene_with_void():
    t = _scene(10)
    t[14:18, 10:15] = 255  # 4x5 void blob inside water
    return t


def test_boundary_f1_ignores_void_blob_boundary():
    t = _scene_with_void()
    perfect = t.clone()
    perfect[14:18, 10:15] = 1  # a real model must label the void with some class
    assert boundary_f1(perfect, t, class_id=1, tol=2) == 1.0

    invents_obstacle = t.clone()
    invents_obstacle[14:18, 10:15] = 0  # invents an obstacle inside the void
    assert boundary_f1(invents_obstacle, t, class_id=1, tol=2) == 1.0
