import json

import numpy as np

from vigie.data import Track, Video, _clean, load_cache, load_split, save_cache


def _kp(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=51).round(2).tolist()


def test_load_split_and_cache_roundtrip(tmp_path):
    pose = tmp_path / "pose"
    gt = tmp_path / "gt"
    pose.mkdir()
    gt.mkdir()
    doc = {"1": {"0": {"keypoints": _kp(0)}, "2": {"keypoints": _kp(1)}},
           "7": {"1": {"keypoints": _kp(2)}}}  # fmt: skip
    (pose / "03_0001.json").write_text(json.dumps(doc))
    np.save(gt / "03_0001.npy", np.array([0, 1, 1, 0], dtype=np.int64))

    videos = load_split(pose, gt)
    assert len(videos) == 1
    v = videos[0]
    assert v.camera == "3" and v.n_frames == 4
    assert [t.pid for t in v.tracks] == ["1", "7"]
    assert v.tracks[0].frames.tolist() == [0, 2]
    assert v.tracks[0].kps.shape == (2, 17, 3)

    save_cache(videos, tmp_path / "c.npz")
    back = load_cache(tmp_path / "c.npz")
    assert back[0].name == "03_0001" and back[0].labels.tolist() == [0, 1, 1, 0]
    assert [t.pid for t in back[0].tracks] == ["1", "7"]
    np.testing.assert_allclose(back[0].tracks[0].kps, v.tracks[0].kps, atol=0.05)


def test_clean_fills_nan_from_neighbours():
    kps = np.ones((3, 17, 3), dtype=np.float32)
    kps[1, 4, 0] = np.nan
    kps[2, 4, :] = 5.0
    out = _clean(kps)
    assert np.isfinite(out).all()
    assert out[1, 4, 2] == 0.0  # joint marked unreliable
    assert out[1, 4, 0] == 5.0 or out[1, 4, 0] == 1.0  # copied from a neighbour


def test_cache_handles_video_without_tracks(tmp_path):
    vids = [Video("01_0001", "1", [], np.zeros(3, np.uint8), 3),
            Video("01_0002", "1", [Track("a", np.arange(2, dtype=np.int32),
                                          np.zeros((2, 17, 3), np.float32))], None, 2)]  # fmt: skip
    save_cache(vids, tmp_path / "c.npz")
    back = load_cache(tmp_path / "c.npz")
    assert back[0].tracks == [] and back[0].labels.tolist() == [0, 0, 0]
    assert len(back[1].tracks) == 1 and back[1].labels is None
