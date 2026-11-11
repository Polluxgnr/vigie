import numpy as np

from vigie.data import Track
from vigie.geometry import fill_gaps, flip_lr, normalise, window_starts


def standing_pose(scale=100.0, angle_deg=0.0, offset=(500.0, 300.0)) -> np.ndarray:
    """A simple upright skeleton (y down), optionally rotated, scaled and shifted."""
    pts = np.zeros((17, 2))
    pts[0] = (0, -1.6)  # nose
    pts[1:5] = [(-0.1, -1.7), (0.1, -1.7), (-0.2, -1.6), (0.2, -1.6)]
    pts[5], pts[6] = (-0.3, -1.0), (0.3, -1.0)  # shoulders
    pts[7], pts[8] = (-0.35, -0.5), (0.35, -0.5)
    pts[9], pts[10] = (-0.35, 0.0), (0.35, 0.0)  # wrists at hip height
    pts[11], pts[12] = (-0.2, 0.0), (0.2, 0.0)  # hips
    pts[13], pts[14] = (-0.2, 0.9), (0.2, 0.9)
    pts[15], pts[16] = (-0.2, 1.8), (0.2, 1.8)
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = pts @ rot.T * scale + np.array(offset)
    return np.concatenate([xy, np.ones((17, 1))], axis=1).astype(np.float32)


def test_normalise_cancels_rotation_scale_and_position():
    ref = normalise(np.stack([standing_pose()] * 5))
    for angle, scale in [(90, 40), (-135, 250), (180, 70)]:
        out = normalise(np.stack([standing_pose(scale, angle, (10, 900))] * 5))
        np.testing.assert_allclose(out[..., :2], ref[..., :2], atol=1e-4)
    # torso has unit length and points up (-y)
    torso = (ref[0, 5, :2] + ref[0, 6, :2]) / 2 - (ref[0, 11, :2] + ref[0, 12, :2]) / 2
    np.testing.assert_allclose(torso, [0, -1], atol=1e-6)


def test_anchor_first_keeps_displacement():
    seq = np.stack([standing_pose(offset=(500 + 10 * t, 300)) for t in range(6)])
    moving = normalise(seq, anchor="first")
    still = normalise(seq, anchor="frame")
    assert np.abs(still[:, 11, :2] + still[:, 12, :2]).max() < 1e-4  # mid-hip at origin
    hips = (moving[:, 11, :2] + moving[:, 12, :2]) / 2
    assert abs(hips[-1, 0] - hips[0, 0]) > 0.4  # walked about half a torso length


def test_fill_gaps_interpolates_small_and_splits_large():
    kps = np.stack([standing_pose(offset=(t, 0)) for t in range(4)])
    tr = Track("a", np.array([0, 2, 3, 20], dtype=np.int32), kps)
    segs = fill_gaps(tr, max_gap=5)
    assert [s[0].tolist() for s in segs] == [[0, 1, 2, 3], [20]]
    filled = segs[0][1]
    assert filled[1, 0, 2] == 0.0  # interpolated joint flagged as unobserved
    np.testing.assert_allclose(filled[1, 0, 0], (kps[0, 0, 0] + kps[1, 0, 0]) / 2)


def test_window_starts_cover_end():
    assert window_starts(10, 24, 6).tolist() == []
    assert window_starts(24, 24, 6).tolist() == [0]
    assert window_starts(33, 24, 6).tolist() == [0, 6, 9]


def test_flip_is_involution():
    p = normalise(np.stack([standing_pose()] * 2))
    np.testing.assert_allclose(flip_lr(flip_lr(p)), p)
