import numpy as np

from vidmcp.perception.mask_ops import feather_mask, iou, temporal_stability_score, to_u8_mask


def test_to_u8_and_feather():
    m = np.zeros((32, 32), dtype=np.float32)
    m[8:24, 8:24] = 1.0
    u = to_u8_mask(m)
    assert u.max() == 255
    f = feather_mask(u, radius=2)
    assert f.shape == u.shape
    assert f.dtype == np.uint8


def test_stability():
    a = np.zeros((16, 16), dtype=np.uint8)
    a[4:12, 4:12] = 255
    b = a.copy()
    b[4:12, 5:13] = 255
    assert iou(a, a) == 1.0
    assert 0.5 < iou(a, b) < 1.0
    assert temporal_stability_score([a, a, a]) > 0.99
