"""Tests for loss weight scheduling functionality.

Tests the ramped topology-loss scheduler implementation including:
- Schedule configuration parsing from YAML
- Weight computation at various epochs
- ThinManifoldLoss.update_weights() functionality
- LossWeightSchedulerCallback integration
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from approach.unetbasic.train import LossWeightSchedulerCallback
from approach.unetbasic.utils.loss_weight_schedule import compute_scheduled_weights
from approach.unetbasic.utils.losses import ThinManifoldLoss


class TestLossWeightScheduleParsing:
    """Test schedule configuration parsing from YAML."""

    def test_load_exp_v0_schedule(self):
        """Test loading loss_weight_schedule from exp_v0.yaml."""
        config_path = project_root / "src/approach/unetbasic/configs/experiments/exp_v0.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        cfg = OmegaConf.load(config_path)
        assert hasattr(cfg, "loss_weight_schedule"), "loss_weight_schedule missing from config"
        assert cfg.loss_weight_schedule.enabled is True, "Schedule should be enabled"
        assert hasattr(cfg.loss_weight_schedule, "ramps"), "ramps section missing"

        # Verify expected ramps are present
        ramps = cfg.loss_weight_schedule.ramps
        expected_weights = [
            "dice_weight",
            "bce_weight",
            "tversky_weight",
            "surface_weight",
            "topo_weight",
            "connectivity_weight",
            "boundary_weight",
        ]
        for weight in expected_weights:
            assert hasattr(ramps, weight), f"Missing ramp for {weight}"
            ramp = getattr(ramps, weight)
            assert hasattr(ramp, "start_epoch"), f"{weight}: missing start_epoch"
            assert hasattr(ramp, "end_epoch"), f"{weight}: missing end_epoch"
            assert hasattr(ramp, "start_value"), f"{weight}: missing start_value"
            assert hasattr(ramp, "end_value"), f"{weight}: missing end_value"

    def test_schedule_structure_validation(self):
        """Test that schedule has correct structure."""
        config_path = project_root / "src/approach/unetbasic/configs/experiments/exp_v0.yaml"
        cfg = OmegaConf.load(config_path)
        ramps = cfg.loss_weight_schedule.ramps

        # Verify non-negative values
        for weight_name in [
            "dice_weight",
            "bce_weight",
            "tversky_weight",
            "surface_weight",
            "topo_weight",
            "connectivity_weight",
            "boundary_weight",
        ]:
            ramp = getattr(ramps, weight_name)
            assert ramp.start_value >= 0, f"{weight_name}: start_value must be non-negative"
            assert ramp.end_value >= 0, f"{weight_name}: end_value must be non-negative"
            assert ramp.start_epoch >= 0, f"{weight_name}: start_epoch must be non-negative"
            assert ramp.end_epoch >= ramp.start_epoch, (
                f"{weight_name}: end_epoch must be >= start_epoch"
            )


class TestWeightComputation:
    """Test weight computation at various epochs using compute_scheduled_weights."""

    def create_test_schedule(self):
        """Create a test schedule config."""
        return OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "dice_weight": {
                        "start_epoch": 0,
                        "end_epoch": 0,
                        "start_value": 0.15,
                        "end_value": 0.15,
                    },
                    "surface_weight": {
                        "start_epoch": 0,
                        "end_epoch": 10,
                        "start_value": 0.05,
                        "end_value": 0.15,
                    },
                    "topo_weight": {
                        "start_epoch": 10,
                        "end_epoch": 40,
                        "start_value": 0.0,
                        "end_value": 0.15,
                    },
                    "connectivity_weight": {
                        "start_epoch": 10,
                        "end_epoch": 40,
                        "start_value": 0.0,
                        "end_value": 0.10,
                    },
                },
            }
        )

    def test_weight_computation_at_epoch_0(self):
        """Test weight computation at epoch 0 (start of training)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=0.0)

        assert "dice_weight" in weights
        assert weights["dice_weight"] == 0.15, "dice_weight should be constant at 0.15"

        assert "surface_weight" in weights
        assert weights["surface_weight"] == 0.05, "surface_weight should start at 0.05"

        assert "topo_weight" in weights
        assert weights["topo_weight"] == 0.0, "topo_weight should start at 0.0"

    def test_weight_computation_mid_ramp(self):
        """Test weight computation during ramp (epoch 5 for surface_weight)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=5.0)

        # surface_weight ramps from 0.05 to 0.15 over epochs 0-10
        # At epoch 5 (midpoint): 0.05 + (0.15 - 0.05) * (5/10) = 0.10
        assert "surface_weight" in weights
        assert abs(weights["surface_weight"] - 0.10) < 1e-6, (
            f"Expected 0.10, got {weights['surface_weight']}"
        )

    def test_weight_computation_at_epoch_10(self):
        """Test weight computation at epoch 10 (ramp transition)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=10.0)

        # surface_weight completes ramp
        assert weights["surface_weight"] == 0.15, "surface_weight should reach 0.15 at epoch 10"

        # topo_weight starts ramping
        assert weights["topo_weight"] == 0.0, "topo_weight should start at 0.0 at epoch 10"

    def test_weight_computation_at_epoch_25(self):
        """Test weight computation during topology ramp (epoch 25)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=25.0)

        # topo_weight ramps from 0.0 to 0.15 over epochs 10-40
        # At epoch 25: 0.0 + (0.15 - 0.0) * ((25-10)/(40-10)) = 0.15 * 0.5 = 0.075
        assert abs(weights["topo_weight"] - 0.075) < 1e-6, (
            f"Expected 0.075, got {weights['topo_weight']}"
        )

        # connectivity_weight ramps from 0.0 to 0.10 over epochs 10-40
        # At epoch 25: 0.0 + (0.10 - 0.0) * 0.5 = 0.05
        assert abs(weights["connectivity_weight"] - 0.05) < 1e-6, (
            f"Expected 0.05, got {weights['connectivity_weight']}"
        )

    def test_weight_computation_after_ramp(self):
        """Test weight computation after all ramps complete (epoch 100)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=100.0)

        # All weights should be at their end values
        assert weights["dice_weight"] == 0.15
        assert weights["surface_weight"] == 0.15
        assert weights["topo_weight"] == 0.15
        assert weights["connectivity_weight"] == 0.10

    def test_fractional_epoch(self):
        """Test weight computation with fractional epoch (e.g., 5.5)."""
        schedule = self.create_test_schedule()
        weights = compute_scheduled_weights(schedule, epoch=5.5)

        # surface_weight at epoch 5.5: 0.05 + (0.15 - 0.05) * (5.5/10) = 0.105
        expected = 0.05 + 0.10 * 0.55
        assert abs(weights["surface_weight"] - expected) < 1e-6


class TestThinManifoldLossUpdateWeights:
    """Test ThinManifoldLoss.update_weights() method."""

    def test_update_single_weight(self):
        """Test updating a single weight."""
        loss_fn = ThinManifoldLoss(
            dice_weight=0.2,
            bce_weight=0.5,
            tversky_weight=0.3,
            topo_weight=0.0,
        )

        # Update topo_weight
        result = loss_fn.update_weights({"topo_weight": 0.15})

        assert loss_fn.topo_weight == 0.15, "topo_weight should be updated"
        assert "topo_weight" in result
        assert result["topo_weight"] == 0.15

        # Other weights should remain unchanged
        assert loss_fn.dice_weight == 0.2
        assert loss_fn.bce_weight == 0.5

    def test_update_multiple_weights(self):
        """Test updating multiple weights simultaneously."""
        loss_fn = ThinManifoldLoss(
            dice_weight=0.15,
            bce_weight=0.15,
            surface_weight=0.05,
            topo_weight=0.0,
            connectivity_weight=0.0,
        )

        updates = {
            "surface_weight": 0.10,
            "topo_weight": 0.075,
            "connectivity_weight": 0.05,
        }
        result = loss_fn.update_weights(updates)

        assert loss_fn.surface_weight == 0.10
        assert loss_fn.topo_weight == 0.075
        assert loss_fn.connectivity_weight == 0.05

        # Verify return dict contains all weights
        assert len(result) == 8  # All 8 weights
        assert result["surface_weight"] == 0.10
        assert result["topo_weight"] == 0.075

    def test_update_unknown_weight(self):
        """Test updating a non-existent weight (should log warning but not crash)."""
        loss_fn = ThinManifoldLoss()

        # This should not raise an error
        result = loss_fn.update_weights({"unknown_weight": 0.5})

        # Result should still contain all valid weights
        assert "dice_weight" in result
        assert "unknown_weight" not in result

    def test_update_returns_all_weights(self):
        """Test that update_weights returns all current weight values."""
        loss_fn = ThinManifoldLoss(
            dice_weight=0.15,
            bce_weight=0.15,
            tversky_weight=0.20,
            surface_weight=0.10,
            topo_weight=0.075,
            connectivity_weight=0.05,
            boundary_weight=0.10,
        )

        result = loss_fn.update_weights({"topo_weight": 0.15})

        # Should return all 8 weights
        expected_keys = {
            "dice_weight",
            "bce_weight",
            "tversky_weight",
            "surface_weight",
            "topo_weight",
            "connectivity_weight",
            "boundary_weight",
            "bifurcation_weight",
        }
        assert set(result.keys()) == expected_keys

        # Verify values
        assert result["dice_weight"] == 0.15
        assert result["bce_weight"] == 0.15
        assert result["tversky_weight"] == 0.20
        assert result["topo_weight"] == 0.15  # Updated value


class TestLossWeightSchedulerCallback:
    """Test LossWeightSchedulerCallback integration."""

    def create_mock_model(self):
        """Create a mock model with ThinManifoldLoss."""

        # Use a simple object instead of Mock to avoid Mock's attribute access issues
        class SimpleModel:
            def __init__(self):
                self.loss_fn = ThinManifoldLoss(
                    dice_weight=0.15,
                    bce_weight=0.15,
                    tversky_weight=0.20,
                    surface_weight=0.05,
                    topo_weight=0.0,
                    connectivity_weight=0.0,
                    boundary_weight=0.0,
                )

        return SimpleModel()

    def create_mock_trainer_state(self, epoch):
        """Create a mock TrainerState with specific epoch."""
        state = Mock()
        state.epoch = epoch
        state.global_step = int(epoch * 100)  # Arbitrary step count
        return state

    def test_callback_initialization(self):
        """Test callback initialization with schedule config."""
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "topo_weight": {
                        "start_epoch": 10,
                        "end_epoch": 40,
                        "start_value": 0.0,
                        "end_value": 0.15,
                    },
                },
            }
        )

        model = self.create_mock_model()
        callback = LossWeightSchedulerCallback(schedule, model)

        assert callback.schedule_cfg == schedule
        assert callback.model == model
        assert callback._last_applied_epoch == -1

    def test_callback_updates_weights_at_epoch_begin(self):
        """Test that callback updates weights on_epoch_begin."""
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "topo_weight": {
                        "start_epoch": 0,
                        "end_epoch": 10,
                        "start_value": 0.0,
                        "end_value": 0.10,
                    },
                },
            }
        )

        model = self.create_mock_model()
        # Store reference to actual loss_fn before callback modifies it
        loss_fn = model.loss_fn
        callback = LossWeightSchedulerCallback(schedule, model)

        # Simulate epoch 5
        state = self.create_mock_trainer_state(epoch=5.0)
        args = Mock()
        control = Mock()

        callback.on_epoch_begin(args, state, control)

        # At epoch 5: 0.0 + (0.10 - 0.0) * 0.5 = 0.05
        # Check the actual loss_fn object we stored earlier
        assert abs(loss_fn.topo_weight - 0.05) < 1e-6

    def test_callback_skips_duplicate_epoch(self):
        """Test that callback doesn't update weights twice for same epoch."""
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "topo_weight": {
                        "start_epoch": 0,
                        "end_epoch": 10,
                        "start_value": 0.0,
                        "end_value": 0.10,
                    },
                },
            }
        )

        model = self.create_mock_model()
        callback = LossWeightSchedulerCallback(schedule, model)

        state = self.create_mock_trainer_state(epoch=5.0)
        args = Mock()
        control = Mock()

        # First call should update
        callback.on_epoch_begin(args, state, control)
        loss_fn = callback._get_loss_fn()
        first_value = loss_fn.topo_weight

        # Manually change weight to detect if callback updates again
        loss_fn.topo_weight = 0.99

        # Second call with same epoch should skip
        callback.on_epoch_begin(args, state, control)
        assert loss_fn.topo_weight == 0.99, "Weight should not be updated again"

    def test_callback_handles_ddp_wrapped_model(self):
        """Test that callback correctly unwraps DDP model."""
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "topo_weight": {
                        "start_epoch": 0,
                        "end_epoch": 10,
                        "start_value": 0.0,
                        "end_value": 0.10,
                    },
                },
            }
        )

        # Create DDP-wrapped model (model.module contains actual model)
        inner_model = self.create_mock_model()
        loss_fn = inner_model.loss_fn  # Store reference to actual loss function

        # Wrap in a simple DDP-like wrapper
        class DDPWrapper:
            def __init__(self, module):
                self.module = module

        wrapped_model = DDPWrapper(inner_model)

        callback = LossWeightSchedulerCallback(schedule, wrapped_model)

        state = self.create_mock_trainer_state(epoch=5.0)
        args = Mock()
        control = Mock()

        callback.on_epoch_begin(args, state, control)

        # Should update inner model's loss function
        assert abs(loss_fn.topo_weight - 0.05) < 1e-6

    def test_callback_with_no_scheduled_weights(self):
        """Test callback when no weights are scheduled for current epoch."""
        # Schedule that only updates at epoch 50+
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "topo_weight": {
                        "start_epoch": 50,
                        "end_epoch": 60,
                        "start_value": 0.0,
                        "end_value": 0.15,
                    },
                },
            }
        )

        model = self.create_mock_model()
        loss_fn = model.loss_fn
        initial_weight = loss_fn.topo_weight
        callback = LossWeightSchedulerCallback(schedule, model)

        # At epoch 5, nothing should change (ramp hasn't started)
        state = self.create_mock_trainer_state(epoch=5.0)
        args = Mock()
        control = Mock()

        callback.on_epoch_begin(args, state, control)

        # Weight should remain at initial value
        assert loss_fn.topo_weight == initial_weight


class TestEndToEndSchedule:
    """Test complete schedule progression over multiple epochs."""

    def test_full_curriculum_schedule(self):
        """Test a full curriculum schedule from epoch 0 to 100."""
        schedule = OmegaConf.create(
            {
                "enabled": True,
                "ramps": {
                    "dice_weight": {
                        "start_epoch": 0,
                        "end_epoch": 0,
                        "start_value": 0.15,
                        "end_value": 0.15,
                    },
                    "surface_weight": {
                        "start_epoch": 0,
                        "end_epoch": 10,
                        "start_value": 0.05,
                        "end_value": 0.15,
                    },
                    "topo_weight": {
                        "start_epoch": 10,
                        "end_epoch": 40,
                        "start_value": 0.0,
                        "end_value": 0.15,
                    },
                },
            }
        )

        # Use simple model object instead of Mock
        class SimpleModel:
            def __init__(self):
                self.loss_fn = ThinManifoldLoss()

        model = SimpleModel()
        loss_fn = model.loss_fn  # Store reference to actual loss function
        callback = LossWeightSchedulerCallback(schedule, model)

        # Track weight progression
        epochs_to_test = [0, 5, 10, 20, 40, 100]
        expected_surface = [0.05, 0.10, 0.15, 0.15, 0.15, 0.15]
        expected_topo = [0.0, 0.0, 0.0, 0.05, 0.15, 0.15]

        for epoch, exp_surface, exp_topo in zip(epochs_to_test, expected_surface, expected_topo):
            # Reset last_applied_epoch to allow update
            callback._last_applied_epoch = -1

            state = Mock()
            state.epoch = float(epoch)
            state.global_step = epoch * 100
            args = Mock()
            control = Mock()

            callback.on_epoch_begin(args, state, control)

            assert abs(loss_fn.surface_weight - exp_surface) < 1e-6, (
                f"Epoch {epoch}: surface_weight={loss_fn.surface_weight}, expected {exp_surface}"
            )
            assert abs(loss_fn.topo_weight - exp_topo) < 1e-6, (
                f"Epoch {epoch}: topo_weight={loss_fn.topo_weight}, expected {exp_topo}"
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
