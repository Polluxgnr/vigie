import numpy as np

from vigie.data import Track, Video
from vigie.eval import AlertPolicy, frame_scores, metrics
from vigie.fusion import NormalReference
from vigie.windows import WIN, WindowSet


def _ws(rows):
    """rows: list of (video, track, frame0, score)."""
    n = len(rows)
    return WindowSet(
        pose=np.zeros((n, WIN, 17, 3), np.float16),
        cues=np.zeros((n, 56), np.float32),
        frame_cues=np.zeros((n, WIN, 14), np.float16),
        video=np.array([r[0] for r in rows], np.int32),
        track=np.array([r[1] for r in rows], np.int32),
        frame0=np.array([r[2] for r in rows], np.int32),
        label=np.zeros(n, np.float32),
    ), np.array([r[3] for r in rows], np.float32)


def test_frame_scores_take_max_over_people_and_floor_empty_frames():
    tr = Track("a", np.arange(60, dtype=np.int32), np.zeros((60, 17, 3), np.float32))
    v = Video("01_0001", "1", [tr, tr], np.zeros(100, np.uint8), 100)
    ws, s = _ws([(0, 0, 0, 0.1), (0, 1, 0, 0.9), (0, 0, 30, 0.2)])
    out = frame_scores([v], ws, s)[0]
    assert out.shape == (100,)
    assert np.allclose(out[:24], 0.9)  # most suspicious person wins
    assert np.allclose(out[30:54], 0.2)
    assert np.allclose(out[60:], 0.1)  # nobody visible -> lowest score


def test_metrics_perfect_ranking():
    v = Video("01_0001", "1", [], np.array([0, 0, 1, 1], np.uint8), 4)
    m = metrics([v], [np.array([0.1, 0.2, 0.8, 0.9])])
    assert m.auc_roc == 1.0 and m.eer == 0.0


def test_alert_policy_needs_sustained_evidence_and_rearms():
    pol = AlertPolicy(threshold=0.5, min_frames=3, ema=1.0, release=0.8)
    spike = np.r_[np.zeros(5), [1.0], np.zeros(5)]
    assert pol.run(spike) == []  # a single frame is not enough
    sustained = np.r_[np.zeros(3), np.ones(6), np.zeros(4), np.ones(4)]
    assert pol.run(sustained) == [5, 15]  # fires once, re-arms after calm, fires again


def test_normal_reference_percentiles():
    ref = NormalReference(np.arange(1000))
    p = ref.percentile(np.array([-1, 499.5, 5000]))
    assert p[0] < 0.001 and abs(p[1] - 0.5) < 0.01 and p[2] > 0.999
    assert np.all(np.diff(ref.logit(np.array([1.0, 2.0, 3.0]))) > 0)
