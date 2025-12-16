from pathlib import Path
from typing import Callable

import torch

from bergson.config import ScoreConfig
from bergson.score.score_writer import MemmapScoreWriter, ScoreWriter


class Scorer:
    scorer_callback: Callable

    num_scores: int

    writer: ScoreWriter

    device: torch.device

    preconditioner: dict[str, torch.Tensor] | None

    def __init__(
        self,
        path: Path,
        num_items: int,
        query_grads: dict[str, torch.Tensor],
        score_cfg: ScoreConfig,
        device: torch.device,
        dtype: torch.dtype,
        preconditioner: dict[str, torch.Tensor] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.num_items = num_items
        self.preconditioner = preconditioner

        self.scorer_callback = self.build_scorer_callback(
            query_grads,
            score_cfg,
        )

        num_scores = len(query_grads[score_cfg.modules[0]])

        self.writer = MemmapScoreWriter(
            path,
            num_items,
            num_scores,
        )

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        first_grad = next(iter(mod_grads.values()))
        if first_grad.dtype != self.dtype:
            mod_grads = {name: grad.to(self.device) for name, grad in mod_grads.items()}

        scores = self.scorer_callback(mod_grads)
        self.writer(indices, scores)

    def build_scorer_callback(
        self,
        query_grads: dict[str, torch.Tensor],
        score_cfg: ScoreConfig,
    ) -> Callable:
        """Unified scorer builder for all scorer types."""
        # Use float32 for scoring when preconditioner is used (for numerical stability)
        # Otherwise use model dtype
        scoring_dtype = torch.float32 if self.preconditioner is not None else self.dtype

        query_tensor = torch.cat(
            [
                query_grads[m].to(device=self.device, dtype=scoring_dtype)
                for m in score_cfg.modules
            ],
            dim=1,
        )

        @torch.inference_mode()
        def callback(mod_grads: dict[str, torch.Tensor]):
            # Apply H^(-1/2) preconditioner to train grads if available
            if self.preconditioner is not None:
                mod_grads = {
                    m: mod_grads[m].float() @ self.preconditioner[m]
                    for m in score_cfg.modules
                }
            grads = torch.cat([mod_grads[m] for m in score_cfg.modules], dim=1)

            # Ensure consistent dtype for dot product
            if grads.dtype != query_tensor.dtype:
                grads = grads.to(query_tensor.dtype)

            if score_cfg.unit_normalize:
                grads /= grads.norm(dim=1, keepdim=True)

            if score_cfg.score == "nearest":
                all_scores = grads @ query_tensor.T
                return all_scores.max(dim=-1).values

            return grads @ query_tensor.T

        return callback
