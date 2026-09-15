"""Tests for ThinManifoldLoss."""

import unittest

import torch

from src.approach.unetbasic.utils.losses import ThinManifoldLoss

IGNORE_INDEX = 2
NEAR_ZERO_THRESHOLD = 1e-3


class TestThinManifoldLoss(unittest.TestCase):
    """Unit tests for `ThinManifoldLoss`."""

    def test_can_go_near_zero_with_perfect_logits(self) -> None:
        """ThinManifoldLoss should be ~0 for perfect predictions (ignoring label==2)."""
        loss_fn = ThinManifoldLoss()

        batch_size = 2
        depth, height, width = 6, 7, 8

        labels = torch.zeros((batch_size, depth, height, width), dtype=torch.int64)
        labels[:, 2:4, 2:5, 3:7] = 1
        labels[:, :2, :2, :2] = IGNORE_INDEX

        logits = torch.full((batch_size, 1, depth, height, width), -20.0, dtype=torch.float32)
        with torch.no_grad():
            logits[:, 0][labels == 1] = 20.0
            logits[:, 0][labels == IGNORE_INDEX] = 0.0
        logits.requires_grad_()

        loss = loss_fn(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())  # noqa: PT009
        self.assertLess(loss.item(), NEAR_ZERO_THRESHOLD)  # noqa: PT009

        loss.backward()
        self.assertIsNotNone(logits.grad)  # noqa: PT009
        self.assertTrue(torch.isfinite(logits.grad).all().item())  # noqa: PT009
