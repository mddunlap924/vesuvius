import pytest

from src.approach.unetbasic.data.dataset import build_augmentation_transforms


def test_build_augmentation_transforms_respects_apply_flag():
    cfg = {
        "RandFlipd": {"apply": True, "prob": 0.5, "spatial_axis": [1, 2]},
        "RandRotate90d": {"apply": False, "prob": 0.5, "max_k": 3},
    }

    tfs = build_augmentation_transforms(cfg)
    assert len(tfs) == 1
    assert tfs[0].__class__.__name__ == "RandFlipd"


def test_build_augmentation_transforms_unknown_raises():
    cfg = {"NotATransform": {"apply": True}}

    with pytest.raises(ValueError, match=r"Unsupported augmentation transform"):
        build_augmentation_transforms(cfg)
