"""Loss functions for 3D segmentation.

Memory optimized: avoids clone(), uses torch.where(), respects autocast dtype.
"""

import logging
from typing import ClassVar

import torch
from monai.losses import DiceLoss, TverskyLoss
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


# =============================================================================
# Sheet Topology Loss - Strict Edge-to-Edge Constraint
# =============================================================================


class AntiBifurcationLoss(nn.Module):
    """Strict topology loss enforcing edge-to-edge sheet constraints.

    Sheets in the XY plane MUST:
    1. Start and end at XY boundaries (no internal endpoints)
    2. Be smooth along Z (gradual warping allowed, sudden jumps penalized)
    3. Not bifurcate, merge, or have internal discontinuities
    4. Not touch or intersect other sheets

    This loss directly penalizes violations using simple, effective operations:

    1. **Z-Smoothness Loss**: Predictions should change gradually between
       adjacent Z slices. Sudden jumps are penalized, but gradual warping
       is allowed (threshold-based).

    2. **Interior Endpoint Loss**: Detects foreground voxels with only 1
       neighbor in XY (endpoints) and penalizes those not at boundaries.

    3. **Row/Column Continuity Loss**: Each scan-line through the volume
       should have a bounded number of foreground segments (no fragmentation).

    4. **Separation Loss**: Penalizes sheets that are too close together
       (potential merging zones).

    All operations are differentiable and computationally efficient.
    """

    def __init__(
        self,
        # Primary topology weights
        z_consistency_weight: float = 2.0,
        endpoint_weight: float = 1.5,
        continuity_weight: float = 1.0,
        separation_weight: float = 0.5,
        # Configuration
        max_segments_per_line: int = 12,  # Max sheet crossings per scan-line
        min_sheet_separation: int = 3,  # Min gap between sheets in voxels
        # Legacy parameters (kept for backward compatibility, ignored)
        density_weight: float = 0.0,
        junction_weight: float = 1.0,
        thickness_weight: float = 0.0,
        neighborhood_size: int = 11,
        max_thickness: float = 9.0,
        expected_arm_count: float = 2.0,
        multi_scale_sizes: tuple[int, ...] = (11, 15, 21),
        anisotropic: bool = True,
        skeleton_erosion_iters: int = 3,
        gap_penalty_weight: float = 0.5,
        min_sheet_gap: float = 5.0,
        gradient_consistency_weight: float = 0.3,
    ) -> None:
        """Initialize sheet topology loss.

        Args:
            z_consistency_weight: Weight for Z-axis smoothness (penalizes sudden jumps).
            endpoint_weight: Weight for penalizing interior endpoints.
            continuity_weight: Weight for scan-line continuity.
            separation_weight: Weight for sheet separation enforcement.
            max_segments_per_line: Maximum expected foreground segments per line.
            min_sheet_separation: Minimum voxels between distinct sheets.

        """
        super().__init__()
        self.z_consistency_weight = z_consistency_weight
        self.endpoint_weight = endpoint_weight
        self.continuity_weight = continuity_weight
        self.separation_weight = separation_weight
        self.max_segments_per_line = max_segments_per_line
        self.min_sheet_separation = min_sheet_separation

    def _z_smoothness_loss(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize sudden changes in predictions between adjacent Z slices.

        Sheets warp and curve gradually along Z, so small slice-to-slice
        variations are expected. However, large sudden jumps indicate
        discontinuities or fragmentation that violate sheet topology.

        Uses a soft threshold to allow gradual changes while penalizing
        abrupt transitions.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].

        Returns:
            Scalar Z-smoothness loss.

        """
        # Compute absolute difference between adjacent Z slices
        # probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]
        z_diff = (probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]).abs()

        # Valid mask for adjacent slice pairs (both slices must be valid)
        valid_pairs = label_valid[:, :, 1:, :, :] * label_valid[:, :, :-1, :, :]

        # Allow small gradual changes (threshold ~0.1 per slice is reasonable)
        # Only penalize changes exceeding this threshold
        gradual_threshold = 0.15
        excess_change = functional.relu(z_diff - gradual_threshold)

        # Weight by valid pairs
        weighted_penalty = excess_change * valid_pairs

        # Normalize by number of valid slice pairs
        num_valid_pairs = valid_pairs.sum().clamp(min=1.0)

        return weighted_penalty.sum() / num_valid_pairs

    # Keep old name as alias for backward compatibility
    _z_consistency_loss = _z_smoothness_loss

    def _endpoint_loss(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize foreground endpoints not at XY boundaries.

        An endpoint is a foreground voxel with very few neighbors. For
        continuous edge-to-edge sheets, endpoints should ONLY exist at
        the boundaries of the XY plane.

        Uses soft neighbor counting to remain differentiable.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].

        Returns:
            Scalar endpoint penalty loss.

        """
        # Count 8-connected neighbors in XY plane
        neighbor_kernel = torch.ones(
            1,
            1,
            1,
            3,
            3,
            device=probs.device,
            dtype=probs.dtype,
        )
        neighbor_kernel[0, 0, 0, 1, 1] = 0  # Exclude center

        neighbor_sum = functional.conv3d(probs, neighbor_kernel, padding=(0, 1, 1))

        # Endpoint indicator: high probability but few neighbors
        # A voxel on a continuous sheet should have ~2+ neighbors
        # Endpoint has < 1.5 effective neighbors
        endpoint_score = probs * functional.relu(1.5 - neighbor_sum)

        # Create interior mask (1 for interior, 0 at boundaries)
        # Endpoints at boundaries are VALID (that's where sheets should end)
        interior_mask = torch.ones_like(probs)
        boundary_width = 2  # Allow endpoints within 2 voxels of edge
        interior_mask[:, :, :, :boundary_width, :] = 0  # Top rows
        interior_mask[:, :, :, -boundary_width:, :] = 0  # Bottom rows
        interior_mask[:, :, :, :, :boundary_width] = 0  # Left cols
        interior_mask[:, :, :, :, -boundary_width:] = 0  # Right cols

        # Only penalize interior endpoints
        interior_endpoints = endpoint_score * interior_mask * label_valid

        # Normalize by foreground mass
        fg_mass = (probs * label_valid).sum().clamp(min=1.0)

        return interior_endpoints.sum() / fg_mass

    def _endpoint_loss_gt_aware(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize interior endpoints, weighted by disagreement with ground truth.

        Only penalizes endpoints in predictions where:
        1. The endpoint is in the interior (not at XY boundaries)
        2. The prediction disagrees with ground truth (false positive endpoints)

        This avoids penalizing predictions that correctly match GT topology.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].
            label_fg_bin: Ground truth binary labels [B, 1, D, H, W].

        Returns:
            Scalar endpoint penalty loss.

        """
        # If no GT provided, fall back to original behavior
        if label_fg_bin is None:
            return self._endpoint_loss(probs, label_valid)

        # Count 8-connected neighbors in XY plane for both pred and GT
        neighbor_kernel = torch.ones(
            1,
            1,
            1,
            3,
            3,
            device=probs.device,
            dtype=probs.dtype,
        )
        neighbor_kernel[0, 0, 0, 1, 1] = 0  # Exclude center

        pred_neighbor_sum = functional.conv3d(probs, neighbor_kernel, padding=(0, 1, 1))
        gt_neighbor_sum = functional.conv3d(label_fg_bin, neighbor_kernel, padding=(0, 1, 1))

        # Endpoint indicator for predictions: high prob but few neighbors
        pred_endpoint_score = probs * functional.relu(1.5 - pred_neighbor_sum)

        # GT endpoint indicator: where GT has endpoints
        gt_endpoint_score = label_fg_bin * functional.relu(1.5 - gt_neighbor_sum)

        # Create interior mask
        interior_mask = torch.ones_like(probs)
        boundary_width = 2
        interior_mask[:, :, :, :boundary_width, :] = 0
        interior_mask[:, :, :, -boundary_width:, :] = 0
        interior_mask[:, :, :, :, :boundary_width] = 0
        interior_mask[:, :, :, :, -boundary_width:] = 0

        # Only penalize SPURIOUS endpoints in predictions:
        # - Prediction has an endpoint (high pred_endpoint_score)
        # - But GT does NOT have an endpoint there (low gt_endpoint_score)
        # This is a false positive endpoint that breaks topology
        spurious_endpoint = pred_endpoint_score * (1.0 - gt_endpoint_score.clamp(max=1.0))

        # Also penalize MISSING endpoints:
        # - GT has an endpoint (high gt_endpoint_score)
        # - But prediction doesn't (low probs or high neighbor count)
        # This means we're predicting continuity where GT has a break
        missing_endpoint = gt_endpoint_score * (1.0 - probs)

        # Combine: penalize topology mismatches in interior
        combined_error = spurious_endpoint + 0.5 * missing_endpoint
        total_endpoint_error = combined_error * interior_mask * label_valid

        # Normalize by foreground mass
        fg_mass = (probs * label_fg_bin).sum().clamp(min=1.0)

        return total_endpoint_error.sum() / fg_mass

    def _continuity_loss(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize fragmented predictions with more segments than ground truth.

        Compares the number of transitions (segment boundaries) in predictions
        vs ground truth along each scan-line. Penalizes only where the prediction
        has MORE transitions than GT, indicating spurious fragmentation.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].
            label_fg_bin: Ground truth binary labels [B, 1, D, H, W].

        Returns:
            Scalar continuity loss.

        """
        # Compute transitions (absolute differences) along X and Y axes
        # Transition = |p[i+1] - p[i]|, high at sheet edges

        # # Valid mask for adjacent pairs
        # valid_x = label_valid[:, :, :, :, 1:] * label_valid[:, :, :, :, :-1]
        # valid_y = label_valid[:, :, :, 1:, :] * label_valid[:, :, :, :-1, :]

        # # Prediction transitions along X (width dimension)
        # pred_trans_x = (probs[:, :, :, :, 1:] - probs[:, :, :, :, :-1]).abs()
        # pred_trans_y = (probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]).abs()

        # Threshold probabilities to hard binary prediction (0/1)
        pred_bin = (probs > 0.5).to(dtype=probs.dtype)

        # Valid mask for adjacent pairs (both voxels must be valid)
        valid_x = label_valid[:, :, :, :, 1:] * label_valid[:, :, :, :, :-1]
        valid_y = label_valid[:, :, :, 1:, :] * label_valid[:, :, :, :-1, :]

        # Prediction transitions along X / Y computed on binary mask (0/1 differences)
        pred_trans_x = (pred_bin[:, :, :, :, 1:] - pred_bin[:, :, :, :, :-1]).abs()
        pred_trans_y = (pred_bin[:, :, :, 1:, :] - pred_bin[:, :, :, :-1, :]).abs()

        # Sum transitions per scan-line for predictions
        pred_trans_per_row = (pred_trans_x * valid_x).sum(dim=4)
        pred_trans_per_col = (pred_trans_y * valid_y).sum(dim=3)

        if label_fg_bin is not None:
            # GT transitions along X and Y
            gt_trans_x = (label_fg_bin[:, :, :, :, 1:] - label_fg_bin[:, :, :, :, :-1]).abs()
            gt_trans_y = (label_fg_bin[:, :, :, 1:, :] - label_fg_bin[:, :, :, :-1, :]).abs()

            # Sum transitions per scan-line for GT
            gt_trans_per_row = (gt_trans_x * valid_x).sum(dim=4)
            gt_trans_per_col = (gt_trans_y * valid_y).sum(dim=3)

            # Penalize where prediction has MORE transitions than GT
            # (spurious fragmentation). Add small tolerance (0.5) to avoid
            # penalizing minor soft-boundary differences
            excess_row = functional.relu(pred_trans_per_row - gt_trans_per_row - 0.5)
            excess_col = functional.relu(pred_trans_per_col - gt_trans_per_col - 0.5)
        else:
            # Fallback: use fixed max transitions if no GT provided
            max_trans = float(self.max_segments_per_line * 2)
            excess_row = functional.relu(pred_trans_per_row - max_trans)
            excess_col = functional.relu(pred_trans_per_col - max_trans)

        # Average penalty across all scan-lines
        n_rows = pred_trans_per_row.numel()
        n_cols = pred_trans_per_col.numel()

        loss_row = excess_row.sum() / max(n_rows, 1)
        loss_col = excess_col.sum() / max(n_cols, 1)

        return (loss_row + loss_col) / 2

    def _separation_loss(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize predictions that merge sheets when GT keeps them separate.

        Uses dilation to detect where foreground regions are within
        min_sheet_separation of each other. Only penalizes if prediction
        shows merging but ground truth does not.

        Strategy:
        1. Dilate predictions to find potential merge zones
        2. Dilate GT to find where GT sheets are close
        3. Penalize only where prediction merges but GT doesn't

        Args:
            probs: Predicted probabilities [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].
            label_fg_bin: Ground truth binary labels [B, 1, D, H, W].

        Returns:
            Scalar separation loss.

        """
        sep = max(3, self.min_sheet_separation)
        if sep % 2 == 0:
            sep += 1

        # Dilate predictions in XY plane
        pred_dilated = functional.max_pool3d(
            probs,
            kernel_size=(1, sep, sep),
            stride=1,
            padding=(0, sep // 2, sep // 2),
        )

        # Double dilation to find merge zones in prediction
        pred_double_dilated = functional.max_pool3d(
            pred_dilated,
            kernel_size=(1, sep, sep),
            stride=1,
            padding=(0, sep // 2, sep // 2),
        )

        # Prediction merge zone: where double dilation exceeds single
        pred_merge_zone = (pred_double_dilated - pred_dilated).clamp(min=0.0)

        if label_fg_bin is not None:
            # Same dilation for GT
            gt_dilated = functional.max_pool3d(
                label_fg_bin,
                kernel_size=(1, sep, sep),
                stride=1,
                padding=(0, sep // 2, sep // 2),
            )
            gt_double_dilated = functional.max_pool3d(
                gt_dilated,
                kernel_size=(1, sep, sep),
                stride=1,
                padding=(0, sep // 2, sep // 2),
            )
            gt_merge_zone = (gt_double_dilated - gt_dilated).clamp(min=0.0)

            # Only penalize where prediction merges but GT doesn't
            # (spurious merging). If GT also has merge zone, it's expected.
            spurious_merge = pred_merge_zone * (1.0 - gt_merge_zone.clamp(max=1.0))
            weighted_penalty = probs * spurious_merge * label_valid
        else:
            # Fallback: penalize all merge zones if no GT provided
            weighted_penalty = probs * pred_merge_zone * label_valid

        # Normalize by foreground mass
        fg_mass = (probs * label_fg_bin).sum().clamp(min=1.0)

        return weighted_penalty.sum() / fg_mass

    def forward(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
        label_fg_bin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute strict sheet topology loss.

        The loss penalizes topology violations in predictions relative to ground truth.
        When label_fg_bin is provided, errors are weighted by disagreement with GT.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W] (after sigmoid).
            label_valid: Valid region mask [B, 1, D, H, W] (1=valid, 0=ignore/mask).
            label_fg_bin: Ground truth binary labels [B, 1, D, H, W] (0=bg, 1=fg).
                         If provided, topology errors are weighted by pred-GT disagreement.

        Returns:
            Scalar topology loss.

        """
        total_loss = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        # Compute disagreement mask: where prediction differs from ground truth
        # This focuses the topology loss on regions where we're making mistakes
        if label_fg_bin is not None:
            # Error regions: high prob but GT=0, or low prob but GT=1
            # Weight topology penalties more heavily in error regions
            error_weight = (probs - label_fg_bin).abs()
            # Combine with base weight (still penalize topology errors even if matching GT)
            # Use 0.3 base + 0.7 error-weighted to focus on mistakes
            topo_weight_mask = (0.3 + 0.7 * error_weight) * label_valid
        else:
            topo_weight_mask = label_valid

        # 1. Z-smoothness (sheets warp gradually, penalize sudden jumps)
        if self.z_consistency_weight > 0:
            loss_z = self._z_smoothness_loss(probs, topo_weight_mask)
            total_loss = total_loss + self.z_consistency_weight * loss_z

        # 2. Endpoint penalty (no internal endpoints)
        # Only penalize endpoints where prediction disagrees with GT
        if self.endpoint_weight > 0:
            loss_endpoint = self._endpoint_loss_gt_aware(
                probs,
                label_valid,
                label_fg_bin,
            )
            total_loss = total_loss + self.endpoint_weight * loss_endpoint

        # 3. Continuity (compare segment count to GT)
        if self.continuity_weight > 0:
            loss_cont = self._continuity_loss(probs, label_valid, label_fg_bin)
            total_loss = total_loss + self.continuity_weight * loss_cont

        # 4. Separation (penalize merges not in GT)
        if self.separation_weight > 0:
            loss_sep = self._separation_loss(probs, label_valid, label_fg_bin)
            total_loss = total_loss + self.separation_weight * loss_sep

        return total_loss


class ThinManifoldLoss(nn.Module):
    """Combined loss for thin 2D manifolds with preprocessed inputs.

    Optimized for single-channel binary segmentation output.
    Expects preprocessed inputs from dataset:
    - label_fg: binary foreground mask (0 or 1) as float32
    - label_valid: valid region mask (1=valid, 0=ignore) as float32

    Components:
    - Dice: Global overlap metric
    - BCE: Voxel-wise binary cross-entropy
    - Tversky: Recall-biased overlap (helpful for very thin positives)
    - Anti-bifurcation: Penalizes junctions/splits where sheets merge or branch

    Memory optimized: no masking logic, just pure math on preprocessed tensors.
    """

    def __init__(
        self,
        dice_weight: float = 0.2,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.3,
        surface_weight: float = 0.0,
        # Topology/connectivity auxiliaries (all optional; set weights > 0 to enable)
        topo_weight: float = 0.0,
        connectivity_weight: float = 0.0,
        boundary_weight: float = 0.0,
        # Anti-bifurcation loss (penalizes junctions/merges)
        bifurcation_weight: float = 0.0,
        bifurcation_density_weight: float = 0.0,  # Disabled by default for thick sheets
        bifurcation_junction_weight: float = 1.0,
        bifurcation_thickness_weight: float = 0.0,  # Disabled by default
        bifurcation_neighborhood_size: int = 11,  # Larger for variable-thickness sheets
        bifurcation_max_thickness: float = 9.0,  # Updated for 3-9 voxel thick sheets
        # New anti-bifurcation parameters for thick sheets
        bifurcation_skeleton_erosion_iters: int = 3,
        bifurcation_gap_weight: float = 0.5,
        bifurcation_min_sheet_gap: float = 5.0,
        bifurcation_gradient_weight: float = 0.3,
        # Topology parameters
        topo_kernel_sizes: tuple[int, ...] | list[int] = (3, 5, 7),
        topo_anisotropic: bool = True,
        topo_closing_frac: float = 0.75,
        topo_opening_frac: float = 0.25,
        connectivity_axis: int = 2,
        connectivity_power: float = 1.0,
        tversky_alpha: float = 0.4,
        tversky_beta: float = 0.6,
    ) -> None:
        """Initialize the loss components.

        Args:
            dice_weight: Weight for Dice loss.
            bce_weight: Weight for binary cross-entropy loss.
            tversky_weight: Weight for Tversky loss.
            surface_weight: Weight for Surface loss (boundary distance-based).
            bifurcation_weight: Weight for anti-bifurcation loss (penalizes junctions).
            bifurcation_density_weight: Internal weight for density penalty (0 to disable).
            bifurcation_junction_weight: Internal weight for skeleton junction detection.
            bifurcation_thickness_weight: Internal weight for thickness constraint (0 to disable).
            bifurcation_neighborhood_size: Size of neighborhood for density analysis.
            bifurcation_max_thickness: Expected maximum thickness of sheets (3-9 voxels).
            bifurcation_skeleton_erosion_iters: Iterations to approximate skeleton.
            bifurcation_gap_weight: Weight for inter-sheet gap penalty.
            bifurcation_min_sheet_gap: Minimum expected gap between sheets.
            bifurcation_gradient_weight: Weight for gradient consistency penalty.
            tversky_alpha: Tversky alpha (FP penalty).
            tversky_beta: Tversky beta (FN penalty).

        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.surface_weight = surface_weight
        self.topo_weight = topo_weight
        self.connectivity_weight = connectivity_weight
        self.boundary_weight = boundary_weight
        self.bifurcation_weight = bifurcation_weight
        # Store anti-bifurcation params for reference
        self._bifurcation_density_weight = bifurcation_density_weight
        self._bifurcation_junction_weight = bifurcation_junction_weight
        self._bifurcation_thickness_weight = bifurcation_thickness_weight
        self._bifurcation_neighborhood_size = bifurcation_neighborhood_size
        self._bifurcation_max_thickness = bifurcation_max_thickness
        self._bifurcation_skeleton_erosion_iters = bifurcation_skeleton_erosion_iters
        self._bifurcation_gap_weight = bifurcation_gap_weight
        self._bifurcation_min_sheet_gap = bifurcation_min_sheet_gap
        self._bifurcation_gradient_weight = bifurcation_gradient_weight
        # OmegaConf commonly materializes tuples as lists; normalize to tuple.
        self.topo_kernel_sizes = tuple(int(k) for k in topo_kernel_sizes)
        self.topo_anisotropic = topo_anisotropic
        self.topo_closing_frac = topo_closing_frac
        self.topo_opening_frac = topo_opening_frac
        self.connectivity_axis = connectivity_axis
        self.connectivity_power = connectivity_power

        if (
            dice_weight < 0
            or bce_weight < 0
            or tversky_weight < 0
            or surface_weight < 0
            or topo_weight < 0
            or connectivity_weight < 0
            or boundary_weight < 0
            or bifurcation_weight < 0
        ):
            msg = "Loss weights must be non-negative"
            raise ValueError(msg)
        if (
            dice_weight
            + bce_weight
            + tversky_weight
            + surface_weight
            + topo_weight
            + connectivity_weight
            + boundary_weight
            + bifurcation_weight
            <= 0
        ):
            msg = "At least one loss weight must be > 0"
            raise ValueError(msg)

        if not topo_kernel_sizes:
            msg = "topo_kernel_sizes must be non-empty"
            raise ValueError(msg)
        if any((k < 1 or k % 2 == 0) for k in topo_kernel_sizes):
            msg = "topo_kernel_sizes must be odd integers >= 1"
            raise ValueError(msg)
        if topo_closing_frac < 0 or topo_opening_frac < 0:
            msg = "topo_closing_frac/topo_opening_frac must be non-negative"
            raise ValueError(msg)
        if topo_closing_frac + topo_opening_frac <= 0 and topo_weight > 0:
            msg = "topo_closing_frac + topo_opening_frac must be > 0 when topo_weight > 0"
            raise ValueError(msg)
        if connectivity_axis not in (2, 3, 4):
            msg = "connectivity_axis must be one of {2 (D/Z), 3 (H/Y), 4 (W/X)} for [B,1,D,H,W] tensors"
            raise ValueError(msg)
        if connectivity_power <= 0:
            msg = "connectivity_power must be > 0"
            raise ValueError(msg)

        # Apply sigmoid manually before MONAI losses for proper masking
        self.dice = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=False,  # We apply sigmoid manually
            reduction="mean",
        )

        self.tversky = TverskyLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=False,  # We apply sigmoid manually
            reduction="mean",
            alpha=tversky_alpha,
            beta=tversky_beta,
        )

        # Anti-bifurcation loss to penalize junctions/merges
        # Uses skeleton-based junction detection for variable-thickness sheets
        self.anti_bifurcation = AntiBifurcationLoss(
            density_weight=self._bifurcation_density_weight,
            junction_weight=self._bifurcation_junction_weight,
            thickness_weight=self._bifurcation_thickness_weight,
            neighborhood_size=self._bifurcation_neighborhood_size,
            max_thickness=self._bifurcation_max_thickness,
            skeleton_erosion_iters=self._bifurcation_skeleton_erosion_iters,
            gap_penalty_weight=self._bifurcation_gap_weight,
            min_sheet_gap=self._bifurcation_min_sheet_gap,
            gradient_consistency_weight=self._bifurcation_gradient_weight,
            anisotropic=self.topo_anisotropic,
        )

    def update_weights(self, weight_dict: dict[str, float]) -> dict[str, float]:
        """Update loss weights dynamically during training.

        Args:
            weight_dict: Dictionary of weight names and new values to apply.

        Returns:
            Dictionary of all current weight values after update.

        """
        # Apply provided weight updates
        for name, value in weight_dict.items():
            if hasattr(self, name):
                setattr(self, name, float(value))
            else:
                logger.warning(f"Attempted to set unknown weight '{name}' on ThinManifoldLoss")

        # Return current state of all weights for logging
        return {
            "dice_weight": self.dice_weight,
            "bce_weight": self.bce_weight,
            "tversky_weight": self.tversky_weight,
            "surface_weight": self.surface_weight,
            "topo_weight": self.topo_weight,
            "connectivity_weight": self.connectivity_weight,
            "boundary_weight": self.boundary_weight,
            "bifurcation_weight": self.bifurcation_weight,
        }

    @staticmethod
    def _erosion3d(x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
        # Erosion via min-pooling implemented as negative max-pooling
        kz, ky, kx = kernel_size
        return -functional.max_pool3d(
            -x,
            kernel_size=(kz, ky, kx),
            stride=1,
            padding=(kz // 2, ky // 2, kx // 2),
        )

    @staticmethod
    def _dilation3d(x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
        kz, ky, kx = kernel_size
        return functional.max_pool3d(
            x,
            kernel_size=(kz, ky, kx),
            stride=1,
            padding=(kz // 2, ky // 2, kx // 2),
        )

    @classmethod
    def _closing3d(cls, x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
        return cls._erosion3d(cls._dilation3d(x, kernel_size), kernel_size)

    @classmethod
    def _opening3d(cls, x: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
        return cls._dilation3d(cls._erosion3d(x, kernel_size), kernel_size)

    @staticmethod
    def _dice_soft(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred/target/mask: [B,1,D,H,W]
        pred_m = pred * mask
        target_m = target * mask
        inter = (pred_m * target_m).sum()
        denom = (pred_m.sum() + target_m.sum()).clamp(min=1e-6)
        return 1.0 - (2.0 * inter / denom)

    def _boundary_dice_loss(
        self,
        probs: torch.Tensor,
        label_fg_bin: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Morphological gradient as a soft boundary proxy: x - erosion(x)
        # Use anistropic (1,3,3) by default to target sheet boundaries per-slice.
        kernel = (1, 3, 3) if self.topo_anisotropic else (3, 3, 3)
        p_er = self._erosion3d(probs, kernel)
        g_er = self._erosion3d(label_fg_bin, kernel)
        p_b = (probs - p_er).clamp(min=0.0)
        g_b = (label_fg_bin - g_er).clamp(min=0.0)
        return self._dice_soft(p_b, g_b, label_valid)

    def _topology_stability_loss(
        self,
        probs: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Encourage prediction to be stable under small closing/opening.
        # - closing adds voxels to fill small holes/gaps (reduces Betti-1/2 artifacts)
        # - opening removes thin spurs/bridges (reduces spurious handles/merges)
        valid_sum = label_valid.sum().clamp(min=1.0)
        closing_losses: list[torch.Tensor] = []
        opening_losses: list[torch.Tensor] = []

        for k in self.topo_kernel_sizes:
            kernel = (1, k, k) if self.topo_anisotropic else (k, k, k)
            closed = self._closing3d(probs, kernel)
            opened = self._opening3d(probs, kernel)

            # Only penalize directional change (added-by-closing / removed-by-opening)
            added = (closed - probs).clamp(min=0.0)
            removed = (probs - opened).clamp(min=0.0)
            closing_losses.append((added * label_valid).sum() / valid_sum)
            opening_losses.append((removed * label_valid).sum() / valid_sum)

        l_close = (
            torch.stack(closing_losses).mean()
            if closing_losses
            else torch.tensor(0.0, device=probs.device)
        )
        l_open = (
            torch.stack(opening_losses).mean()
            if opening_losses
            else torch.tensor(0.0, device=probs.device)
        )

        frac_sum = self.topo_closing_frac + self.topo_opening_frac
        close_w = self.topo_closing_frac / frac_sum
        open_w = self.topo_opening_frac / frac_sum
        return close_w * l_close + open_w * l_open

    def _connectivity_tv_loss(
        self,
        probs: torch.Tensor,
        label_fg_bin: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        # Weighted anisotropic total-variation along a chosen axis.
        # For sheets in volumes, continuity along Z (axis=2) is often the biggest win.
        axis = self.connectivity_axis
        # Weight where either GT or prediction suggests foreground, and only where valid.
        w = (label_fg_bin + probs).clamp(0.0, 1.0) * label_valid

        # Build slice pairs along the selected axis
        diff = torch.abs(
            probs.narrow(axis, 1, probs.size(axis) - 1)
            - probs.narrow(axis, 0, probs.size(axis) - 1),
        )
        w_pair = (w.narrow(axis, 1, w.size(axis) - 1) + w.narrow(axis, 0, w.size(axis) - 1)).clamp(
            min=0.0,
        )

        # Optional robust power (L1 by default)
        if self.connectivity_power != 1.0:
            diff = diff.clamp(min=1e-6).pow(self.connectivity_power)

        denom = w_pair.sum().clamp(min=1.0)
        return (diff * w_pair).sum() / denom

    def _compute_surface_loss(
        self,
        probs: torch.Tensor,
        distance_map: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surface loss using distance map.

        Surface loss penalizes predictions proportional to their distance
        from the true boundary, encouraging accurate boundary localization.

        Args:
            probs: Predicted probabilities [B, 1, D, H, W]
            distance_map: Distance transform of foreground [B, 1, D, H, W]
            label_valid: Valid region mask [B, 1, D, H, W]

        Returns:
            Scalar surface loss

        """
        # Signed distance field (SDF) is expected:
        #   negative inside the GT foreground, positive outside.
        # Kervadec-style surface loss: mean(probs * sdf).
        # We mask invalid voxels and normalize by valid count.
        sdf_masked = distance_map * label_valid
        surface_loss = (probs * sdf_masked).sum() / label_valid.sum().clamp(min=1.0)

        return surface_loss

    def forward(
        self,
        logits: torch.Tensor,
        label_fg_bin: torch.Tensor,
        label_valid: torch.Tensor | None = None,
        distance_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for thin manifold segmentation.

        Args:
            logits: Raw logits with shape [B, 1, D, H, W].
            label_fg_bin: Binary foreground mask with shape [B, 1, D, H, W], float32.
            label_valid: Valid region mask with shape [B, 1, D, H, W], float32.
            distance_map: Distance transform of foreground with shape [B, 1, D, H, W], float32.
                         Required if surface_weight > 0.

        Returns:
            Scalar loss tensor.

        """
        # Backward-compat: allow calling with a single label tensor where
        # labels are {0=bg, 1=fg, 2=ignore}.
        if label_valid is None:
            labels = label_fg_bin
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)  # [B,D,H,W] -> [B,1,D,H,W]
            # label_valid: 1 where not ignored
            label_valid = (labels != 2).to(dtype=logits.dtype)
            # label_fg_bin: 1 where foreground
            label_fg_bin = (labels == 1).to(dtype=logits.dtype)

        # Check for fully masked batch
        if not label_valid.any():
            logger.warning("All voxels are masked out in this batch.")
            return logits.sum() * 0.0

        # Apply sigmoid and mask in probability space
        probs = torch.sigmoid(logits)
        probs_masked = probs * label_valid
        label_fg_masked = label_fg_bin * label_valid

        # Count valid voxels
        num_valid = label_valid.sum().clamp(min=1.0)

        # Compute individual loss components
        loss_dice = self.dice(probs_masked, label_fg_masked)

        bce_map = functional.binary_cross_entropy_with_logits(
            logits,
            label_fg_bin,
            reduction="none",
        )
        loss_bce = (bce_map * label_valid).sum() / num_valid

        loss_tversky = self.tversky(probs_masked, label_fg_masked)

        # Surface loss (boundary-aware)
        loss_surface = torch.tensor(0.0, device=logits.device)
        if self.surface_weight > 0:
            if distance_map is None:
                msg = "distance_map is required when surface_weight > 0"
                raise ValueError(msg)
            loss_surface = self._compute_surface_loss(probs_masked, distance_map, label_valid)

        # Sheet-friendly topology/connectivity auxiliaries
        loss_topo = torch.tensor(0.0, device=logits.device)
        if self.topo_weight > 0:
            loss_topo = self._topology_stability_loss(probs_masked, label_valid)

        loss_conn = torch.tensor(0.0, device=logits.device)
        if self.connectivity_weight > 0:
            loss_conn = self._connectivity_tv_loss(probs_masked, label_fg_masked, label_valid)

        loss_boundary = torch.tensor(0.0, device=logits.device)
        if self.boundary_weight > 0:
            loss_boundary = self._boundary_dice_loss(probs_masked, label_fg_masked, label_valid)

        # Anti-bifurcation loss (penalizes topology violations relative to ground truth)
        # Passes label_fg_bin so the loss focuses on prediction errors, not arbitrary constraints
        loss_bifurcation = torch.tensor(0.0, device=logits.device)
        if self.bifurcation_weight > 0:
            loss_bifurcation = self.anti_bifurcation(
                probs_masked,
                label_valid,
                label_fg_bin=label_fg_bin,
            )

        # Check for NaN in any component
        if (
            torch.isnan(loss_dice)
            or torch.isnan(loss_bce)
            or torch.isnan(loss_tversky)
            or torch.isnan(loss_surface)
            or torch.isnan(loss_topo)
            or torch.isnan(loss_conn)
            or torch.isnan(loss_boundary)
            or torch.isnan(loss_bifurcation)
        ):
            logger.warning("NaN detected in loss components.")

        # Combine all loss components with their weights
        return (
            self.dice_weight * loss_dice
            + self.bce_weight * loss_bce
            + self.tversky_weight * loss_tversky
            + self.surface_weight * loss_surface
            + self.topo_weight * loss_topo
            + self.connectivity_weight * loss_conn
            + self.boundary_weight * loss_boundary
            + self.bifurcation_weight * loss_bifurcation
        )


class TopologyPreservingLoss(nn.Module):
    """Combined loss for topology-preserving dual-head segmentation.

    Combines classification losses (from ThinManifoldLoss) with distance regression
    losses for improved topological continuity.

    Components:
    1. Classification losses (applied to binary head):
       - Dice, BCE, Tversky, Surface, Topology stability, Connectivity, Boundary

    2. Distance regression losses (applied to distance head):
       - MSE on signed distance field
       - Gradient penalty for smoothness (encourages continuous predictions)

    The distance regression naturally enforces topology because:
    - SDF is inherently smooth and continuous
    - Predicting accurate distances requires understanding sheet geometry
    - Gradient penalty penalizes discontinuities/breaks in predictions
    """

    def __init__(
        self,
        # Classification loss weights (same as ThinManifoldLoss)
        dice_weight: float = 0.15,
        bce_weight: float = 0.15,
        tversky_weight: float = 0.20,
        surface_weight: float = 0.15,
        topo_weight: float = 0.15,
        connectivity_weight: float = 0.10,
        boundary_weight: float = 0.10,
        # Anti-bifurcation loss (penalizes junctions/merges)
        bifurcation_weight: float = 0.0,
        bifurcation_density_weight: float = 0.0,  # Disabled by default for thick sheets
        bifurcation_junction_weight: float = 1.0,
        bifurcation_thickness_weight: float = 0.0,  # Disabled by default
        bifurcation_neighborhood_size: int = 11,  # Larger for variable-thickness sheets
        bifurcation_max_thickness: float = 9.0,  # Updated for 3-9 voxel thick sheets
        # New anti-bifurcation parameters for thick sheets
        bifurcation_skeleton_erosion_iters: int = 3,
        bifurcation_gap_weight: float = 0.5,
        bifurcation_min_sheet_gap: float = 5.0,
        bifurcation_gradient_weight: float = 0.3,
        # Distance regression weights
        distance_weight: float = 0.50,
        gradient_penalty_weight: float = 0.10,
        # Topology parameters (passed to ThinManifoldLoss)
        topo_kernel_sizes: tuple[int, ...] | list[int] = (3, 5, 7),
        topo_anisotropic: bool = True,
        topo_closing_frac: float = 0.75,
        topo_opening_frac: float = 0.25,
        connectivity_axis: int = 2,
        connectivity_power: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        """Initialize topology-preserving loss.

        Args:
            dice_weight: Weight for Dice loss on binary head.
            bce_weight: Weight for BCE loss on binary head.
            tversky_weight: Weight for Tversky loss on binary head.
            surface_weight: Weight for surface loss on binary head.
            topo_weight: Weight for topology stability loss.
            connectivity_weight: Weight for connectivity loss.
            boundary_weight: Weight for boundary Dice loss.
            bifurcation_weight: Weight for anti-bifurcation loss (penalizes junctions).
            bifurcation_density_weight: Internal weight for density penalty (0 to disable).
            bifurcation_junction_weight: Internal weight for skeleton junction detection.
            bifurcation_thickness_weight: Internal weight for thickness constraint (0 to disable).
            bifurcation_neighborhood_size: Size of neighborhood for density analysis.
            bifurcation_max_thickness: Expected maximum thickness of sheets (3-9 voxels).
            bifurcation_skeleton_erosion_iters: Iterations to approximate skeleton.
            bifurcation_gap_weight: Weight for inter-sheet gap penalty.
            bifurcation_min_sheet_gap: Minimum expected gap between sheets.
            bifurcation_gradient_weight: Weight for gradient consistency penalty.
            distance_weight: Weight for MSE on distance regression.
            gradient_penalty_weight: Weight for gradient smoothness penalty.
            topo_kernel_sizes: Kernel sizes for morphological operations.
            topo_anisotropic: Use anisotropic kernels (better for 2D sheets in 3D).
            topo_closing_frac: Fraction of closing in topo loss.
            topo_opening_frac: Fraction of opening in topo loss.
            connectivity_axis: Axis for connectivity loss (2=Z, 3=Y, 4=X).
            connectivity_power: Power for connectivity gradient (1=L1).
            tversky_alpha: Tversky alpha (FP penalty).
            tversky_beta: Tversky beta (FN penalty).

        """
        super().__init__()

        # Store distance regression weights
        self.distance_weight = distance_weight
        self.gradient_penalty_weight = gradient_penalty_weight

        # Validate weights
        if distance_weight < 0 or gradient_penalty_weight < 0:
            msg = "Distance loss weights must be non-negative"
            raise ValueError(msg)

        # Create the classification loss component
        # Note: We'll use it internally but also need access to its components
        self.classification_loss = ThinManifoldLoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            tversky_weight=tversky_weight,
            surface_weight=surface_weight,
            topo_weight=topo_weight,
            connectivity_weight=connectivity_weight,
            boundary_weight=boundary_weight,
            bifurcation_weight=bifurcation_weight,
            bifurcation_density_weight=bifurcation_density_weight,
            bifurcation_junction_weight=bifurcation_junction_weight,
            bifurcation_thickness_weight=bifurcation_thickness_weight,
            bifurcation_neighborhood_size=bifurcation_neighborhood_size,
            bifurcation_max_thickness=bifurcation_max_thickness,
            bifurcation_skeleton_erosion_iters=bifurcation_skeleton_erosion_iters,
            bifurcation_gap_weight=bifurcation_gap_weight,
            bifurcation_min_sheet_gap=bifurcation_min_sheet_gap,
            bifurcation_gradient_weight=bifurcation_gradient_weight,
            topo_kernel_sizes=topo_kernel_sizes,
            topo_anisotropic=topo_anisotropic,
            topo_closing_frac=topo_closing_frac,
            topo_opening_frac=topo_opening_frac,
            connectivity_axis=connectivity_axis,
            connectivity_power=connectivity_power,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )

    def update_weights(self, weight_dict: dict[str, float]) -> dict[str, float]:
        """Update loss weights dynamically during training.

        Args:
            weight_dict: Dictionary of weight names and new values to apply.

        Returns:
            Dictionary of all current weight values after update.

        """
        # Update distance-specific weights
        distance_weight_names = ("distance_weight", "gradient_penalty_weight")
        for name in distance_weight_names:
            if name in weight_dict:
                setattr(self, name, float(weight_dict[name]))

        # Filter out distance-specific weights before delegating to classification loss
        # to avoid "unknown weight" warnings from ThinManifoldLoss
        classification_weight_dict = {
            k: v for k, v in weight_dict.items() if k not in distance_weight_names
        }

        # Delegate classification weights to the inner loss
        classification_weights = self.classification_loss.update_weights(classification_weight_dict)

        # Return combined state
        return {
            **classification_weights,
            "distance_weight": self.distance_weight,
            "gradient_penalty_weight": self.gradient_penalty_weight,
        }

    def _compute_gradient_penalty(
        self,
        distance_pred: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty to encourage smooth distance predictions.

        Penalizes large spatial gradients in the predicted distance field,
        encouraging continuous, topology-preserving predictions.

        For thin sheets, we use anisotropic weighting:
        - Smaller penalty for gradients along Z (depth/scroll axis)
        - Larger penalty for gradients in XY plane (sheet plane)

        Args:
            distance_pred: Predicted distance field [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].

        Returns:
            Scalar gradient penalty loss.

        """
        # Compute spatial gradients along each axis
        # dz: gradient along depth (axis 2)
        # dy: gradient along height (axis 3)
        # dx: gradient along width (axis 4)

        # Use central differences where possible, forward/backward at boundaries
        # For simplicity, use forward differences and valid mask handling

        # Gradient along Z (depth) - axis 2
        dz = torch.abs(
            distance_pred[:, :, 1:, :, :] - distance_pred[:, :, :-1, :, :],
        )
        valid_z = label_valid[:, :, 1:, :, :] * label_valid[:, :, :-1, :, :]

        # Gradient along Y (height) - axis 3
        dy = torch.abs(
            distance_pred[:, :, :, 1:, :] - distance_pred[:, :, :, :-1, :],
        )
        valid_y = label_valid[:, :, :, 1:, :] * label_valid[:, :, :, :-1, :]

        # Gradient along X (width) - axis 4
        dx = torch.abs(
            distance_pred[:, :, :, :, 1:] - distance_pred[:, :, :, :, :-1],
        )
        valid_x = label_valid[:, :, :, :, 1:] * label_valid[:, :, :, :, :-1]

        # Anisotropic weighting: less penalty on Z gradients (sheet extends along Z)
        # This matches the typical data where sheets are thin in XY but extend along Z
        z_weight = 0.5  # Reduce penalty for depth gradients
        xy_weight = 1.0  # Full penalty for in-plane gradients

        # Compute weighted gradient magnitude
        grad_z = (dz * valid_z).sum() / valid_z.sum().clamp(min=1.0)
        grad_y = (dy * valid_y).sum() / valid_y.sum().clamp(min=1.0)
        grad_x = (dx * valid_x).sum() / valid_x.sum().clamp(min=1.0)

        # Combined gradient penalty with anisotropic weighting
        gradient_penalty = z_weight * grad_z + xy_weight * (grad_y + grad_x) / 2

        return gradient_penalty

    def _compute_distance_mse_loss(
        self,
        distance_pred: torch.Tensor,
        distance_target: torch.Tensor,
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss on distance predictions.

        Both prediction and target should be in [-1, 1] range:
        - distance_pred: from model's tanh output, naturally in [-1, 1]
        - distance_target: from ApplyDistanceTransform with asymmetric normalization
          that maps inside distances (thin sheets ~3 voxels) to [-1, 0] and
          outside distances (up to 10 voxels) to [0, 1]

        This asymmetric normalization ensures the model's full tanh range is utilized,
        rather than wasting capacity on the [-1, -0.3] range that would never be used
        with symmetric normalization of thin foreground sheets.

        Args:
            distance_pred: Predicted distance field [B, 1, D, H, W], tanh output in [-1, 1].
            distance_target: Target SDF [B, 1, D, H, W], asymmetrically normalized to [-1, 1].
            label_valid: Valid region mask [B, 1, D, H, W].

        Returns:
            Scalar MSE loss.

        """
        # Compute squared error only in valid regions
        sq_error = (distance_pred - distance_target) ** 2
        masked_sq_error = sq_error * label_valid

        # Mean over valid voxels
        num_valid = label_valid.sum().clamp(min=1.0)
        mse_loss = masked_sq_error.sum() / num_valid

        return mse_loss

    def forward(
        self,
        logits: torch.Tensor,
        distance_pred: torch.Tensor,
        label_fg_bin: torch.Tensor,
        label_valid: torch.Tensor,
        distance_map: torch.Tensor | None = None,
        distance_unsigned: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined topology-preserving loss.

        Args:
            logits: Binary segmentation logits [B, 1, D, H, W].
            distance_pred: Predicted distance field [B, 1, D, H, W], from tanh.
            label_fg_bin: Binary foreground mask [B, 1, D, H, W].
            label_valid: Valid region mask [B, 1, D, H, W].
            distance_map: Signed distance field [B, 1, D, H, W] (for surface loss).
            distance_unsigned: Unsigned distance from foreground [B, 1, D, H, W]
                              (regression target, not currently used - we use SDF instead).

        Returns:
            Scalar combined loss.

        """
        # 1. Classification loss on binary head
        loss_classification = self.classification_loss(
            logits=logits,
            label_fg_bin=label_fg_bin,
            label_valid=label_valid,
            distance_map=distance_map,
        )

        # 2. Distance regression loss on distance head
        loss_distance = torch.tensor(0.0, device=logits.device)
        if self.distance_weight > 0 and distance_map is not None:
            loss_distance = self._compute_distance_mse_loss(
                distance_pred=distance_pred,
                distance_target=distance_map,
                label_valid=label_valid,
            )

        # 3. Gradient penalty for smoothness
        loss_gradient = torch.tensor(0.0, device=logits.device)
        if self.gradient_penalty_weight > 0:
            loss_gradient = self._compute_gradient_penalty(
                distance_pred=distance_pred,
                label_valid=label_valid,
            )

        # Check for NaN
        if (
            torch.isnan(loss_classification)
            or torch.isnan(loss_distance)
            or torch.isnan(loss_gradient)
        ):
            logger.warning("NaN detected in topology-preserving loss components.")

        # Combine all losses
        total_loss = (
            loss_classification
            + self.distance_weight * loss_distance
            + self.gradient_penalty_weight * loss_gradient
        )

        return total_loss
