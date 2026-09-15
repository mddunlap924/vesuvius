"""Test that the dataset, the collator and the trainer agree on keys and shapes."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from approach.unetbasic.data.dataset import PapyrusDataset
from approach.unetbasic.train import UNETTrainer, data_collator

PATCH_SIZE = (16, 32, 32)

# Keys the dataset produces and the collator is expected to stack.
SAMPLE_KEYS = (
    "pixel_values",
    "label_fg_bin",
    "label_fg",
    "label_valid",
    "distance_unsigned",
    "idx",
)
TENSOR_KEYS = SAMPLE_KEYS[:-1]


def write_case(data_dir, image_id="case0001", patch_size=PATCH_SIZE):
    """Write one 3-class image/label pair in the ``train_images``/``train_labels`` layout."""
    (data_dir / "train_images").mkdir(parents=True, exist_ok=True)
    (data_dir / "train_labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    image = (rng.random(patch_size) * 255).astype(np.uint8)
    # 0 = background, 1 = surface, 2 = unlabelled/ignore
    label = rng.integers(0, 3, patch_size).astype(np.uint8)

    np.save(data_dir / f"train_images/{image_id}.npy", image)
    np.save(data_dir / f"train_labels/{image_id}.npy", label)
    return image_id


@pytest.fixture
def sample(tmp_path):
    image_id = write_case(tmp_path)
    dataset = PapyrusDataset(data_dir=tmp_path, image_ids=[image_id], augmentation=False)
    return dataset[0]


class TestDatasetKeys:
    """The dataset must emit exactly the keys the collator consumes."""

    def test_dataset_returns_expected_keys(self, sample):
        for key in SAMPLE_KEYS:
            assert key in sample, f"Expected '{key}' key, got {sorted(sample.keys())}"

    def test_dataset_carries_the_case_identifier(self, sample):
        assert sample["id"] == "case0001"
        assert sample["idx"] == 0

    def test_dataset_does_not_return_legacy_keys(self, sample):
        """'inputs' and 'labels' are the pre-refactor names and must not reappear."""
        assert "inputs" not in sample
        assert "labels" not in sample


class TestDatasetShapes:
    """Image and target tensors must all be channel-first 4D volumes."""

    def test_dataset_output_shapes(self, sample):
        for key in TENSOR_KEYS:
            assert sample[key].shape == (1, *PATCH_SIZE), (
                f"Expected {key} shape (1, {PATCH_SIZE}), got {sample[key].shape}"
            )

    def test_dataset_output_dtypes(self, sample):
        for key in TENSOR_KEYS:
            assert sample[key].dtype == torch.float32, f"{key} should be float32"

    def test_labels_are_mutually_consistent(self, sample):
        """Valid voxels are a superset of foreground voxels; the SDF is bounded to [-1, 1]."""
        foreground = sample["label_fg_bin"].bool()
        assert not (foreground & ~sample["label_valid"].bool()).any()
        assert sample["label_fg"].min() >= -1.0
        assert sample["label_fg"].max() <= 1.0
        assert sample["distance_unsigned"].min() >= 0.0


class TestDataCollator:
    """The collator stacks every dataset key along a new batch dimension."""

    def test_collator_stacks_dataset_output(self, sample):
        result = data_collator([sample, {**sample, "idx": 1}])

        for key in SAMPLE_KEYS:
            assert key in result, f"Expected '{key}' key, got {sorted(result.keys())}"

        for key in TENSOR_KEYS:
            assert result[key].shape == (2, 1, *PATCH_SIZE), f"unexpected shape for {key}"
        assert result["idx"].tolist() == [0, 1]

    def test_collator_key_error_with_wrong_keys(self):
        batch = [{"wrong_key": torch.rand(1, *PATCH_SIZE)}]

        with pytest.raises(KeyError):
            data_collator(batch)


class TestDatasetToCollatorPipeline:
    """End-to-end: dataset output must survive the collator unchanged."""

    def test_dataset_to_collator_pipeline(self, tmp_path):
        image_id = write_case(tmp_path)
        dataset = PapyrusDataset(data_dir=tmp_path, image_ids=[image_id], augmentation=False)

        result = data_collator([dataset[0]])

        assert result["pixel_values"].shape == (1, 1, *PATCH_SIZE)
        assert result["label_fg_bin"].shape == (1, 1, *PATCH_SIZE)
        assert result["idx"].tolist() == [0]


class TestUNETTrainerComputeLoss:
    """UNETTrainer.compute_loss must forward model inputs and drop bookkeeping keys."""

    def make_inputs(self):
        return {
            "pixel_values": torch.rand(1, 1, *PATCH_SIZE),
            "label_fg_bin": torch.randint(0, 2, (1, 1, *PATCH_SIZE)).float(),
            "label_fg": torch.rand(1, 1, *PATCH_SIZE) * 2 - 1,
            "label_valid": torch.ones(1, 1, *PATCH_SIZE),
            "distance_unsigned": torch.rand(1, 1, *PATCH_SIZE),
            "idx": torch.tensor([0]),
            "id": ["case0001"],
        }

    def test_compute_loss_filters_bookkeeping_keys(self):
        mock_model = MagicMock()
        mock_model.return_value = {"loss": torch.tensor(0.5)}
        trainer = UNETTrainer.__new__(UNETTrainer)

        trainer.compute_loss(mock_model, self.make_inputs())

        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        for key in ("pixel_values", "label_fg_bin", "label_valid", "distance_map"):
            assert key in call_kwargs, f"{key} should be passed to the model"
        assert "distance_unsigned" in call_kwargs
        assert "idx" not in call_kwargs, "idx should NOT be passed to the model"
        assert "id" not in call_kwargs, "id should NOT be passed to the model"

    def test_compute_loss_maps_label_fg_to_distance_map(self):
        mock_model = MagicMock()
        mock_model.return_value = {"loss": torch.tensor(0.5)}
        trainer = UNETTrainer.__new__(UNETTrainer)

        inputs = self.make_inputs()
        trainer.compute_loss(mock_model, inputs)

        assert torch.equal(mock_model.call_args[1]["distance_map"], inputs["label_fg"])

    def test_compute_loss_returns_loss_tensor(self):
        expected_loss = torch.tensor(1.23)
        mock_model = MagicMock()
        mock_model.return_value = {"loss": expected_loss}
        trainer = UNETTrainer.__new__(UNETTrainer)

        loss = trainer.compute_loss(mock_model, self.make_inputs())

        assert loss == expected_loss, f"Expected loss {expected_loss}, got {loss}"
