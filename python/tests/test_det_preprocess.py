# -*- encoding: utf-8 -*-
# @Author: Mohammad Hijjawi
import numpy as np
import pytest
from rapidocr.ch_ppocr_det import TextDetector


def get_detector(limit_side_len: int, limit_type: str) -> TextDetector:
    detector = TextDetector.__new__(TextDetector)
    detector.limit_side_len = limit_side_len
    detector.limit_type = limit_type
    detector.mean = [0.5, 0.5, 0.5]
    detector.std = [0.5, 0.5, 0.5]
    detector.postprocess_op = lambda preds, ori_shape: ([], [])
    return detector


@pytest.mark.parametrize(
    "limit_side_len,limit_type,img_hw,expected_hw",
    [
        # issue #740: limit_side_len must be honoured when limit_type is "max"
        (960, "max", (2000, 2000), (960, 960)),
        (960, "max", (1000, 2000), (480, 960)),
        (640, "max", (1200, 800), (640, 416)),
        (960, "max", (320, 640), (320, 640)),
        (736, "min", (500, 1000), (736, 1472)),
        (736, "min", (1000, 800), (992, 800)),
        (736, "min", (800, 1600), (800, 1600)),
    ],
)
def test_det_preprocess_respects_limit_side_len(
    limit_side_len, limit_type, img_hw, expected_hw
):
    detector = get_detector(limit_side_len, limit_type)
    img = np.zeros((*img_hw, 3), dtype=np.uint8)

    det_inputs = []
    detector.session = det_inputs.append

    detector(img)

    assert len(det_inputs) == 1
    assert det_inputs[0].shape == (1, 3, *expected_hw)


def test_det_does_not_probe_omegaconf_mean_std_on_every_call(monkeypatch):
    # mean/std come from the omegaconf config as ListConfig. Passing them to
    # np.array() on every call made NumPy probe ListConfig for __array__ & co.
    # Each probe raised ConfigAttributeError, and with omegaconf<2.4 the
    # resulting reference cycles kept the input image alive until a full gc.
    import rapidocr.ch_ppocr_det.main as det_main
    from omegaconf import ListConfig, OmegaConf

    det_inputs = []
    monkeypatch.setattr(
        det_main, "get_engine", lambda engine_type: lambda cfg: det_inputs.append
    )
    cfg = OmegaConf.create(
        {
            "engine_type": "onnxruntime",
            "limit_side_len": 736,
            "limit_type": "min",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
    )
    detector = TextDetector(cfg)
    detector.postprocess_op = lambda preds, ori_shape: ([], [])

    probed = []
    orig_getattr = ListConfig.__getattr__

    def spy_getattr(self, key):
        probed.append(key)
        return orig_getattr(self, key)

    monkeypatch.setattr(ListConfig, "__getattr__", spy_getattr)
    detector(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(det_inputs) == 1
    assert probed == []
