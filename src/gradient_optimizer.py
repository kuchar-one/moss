"""Gradient-Based Optimization for Pareto Front finding.

Uses scalarization (Batch of weighted sums) to find diverse solutions in parallel via Adam.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from . import config
from .objectives import calc_image_loss, calc_audio_mag_loss


class ParetoManager(nn.Module):
    def __init__(self, encoder, pop_size=30, learning_rate=0.01):
        super().__init__()
        self.encoder = encoder
        self.pop_size = pop_size

        # 1. Initialize Population of Masks (Logits)
        # We optimize in logit space (unbounded) and apply Sigmoid in forward
        # Shape: (Pop, N_Params) -> but easier as (Pop, 1, H, W) for encoder
        self.grid_h = encoder.grid_height
        self.grid_w = encoder.grid_width

        # Initialize randomly (normal dist around 0)
        self.mask_logits = nn.Parameter(
            torch.randn(pop_size, self.grid_h * self.grid_w, device=encoder.device)
            * 0.5
        )

        # 2. Assign Scalarization Weights
        # Linearly space weights from [1, 0] (Pure Image) to [0, 1] (Pure Audio)
        # This forces the population to spread across the front.
        alpha = torch.linspace(0, 1, pop_size, device=encoder.device)
        # w_img goes 1 -> 0
        # w_aud goes 0 -> 1
        self.weights_img = 1.0 - alpha
        self.weights_aud = alpha

        # 3. Optimizer
        self.optimizer = optim.Adam([self.mask_logits], lr=learning_rate)

    def optimize_step(self, image_mag_ref, audio_mag_ref):
        """Performs one step of gradient descent."""
        self.optimizer.zero_grad()

        # 1. Get Masks (Sigmoid to [0,1])
        masks = torch.sigmoid(self.mask_logits)

        # 2. Forward Pass (Batch Evaluation)
        # Encoder takes (B, N_Params)
        audio_recon, mixed_mag = self.encoder(masks)

        # 3. Calculate Losses
        # Visual Loss: (B,)
        loss_vis = calc_image_loss(mixed_mag, image_mag_ref)
        # Audio Loss: (B,)
        loss_aud = calc_audio_mag_loss(mixed_mag, audio_mag_ref)

        # 4. Scalarized Loss (Weighted Sum)
        # We want to minimize: w_img * Vis + w_aud * Aud
        # Sum over batch (since it's parallel independent probs)
        total_loss = torch.sum(
            self.weights_img * loss_vis + self.weights_aud * loss_aud
        )

        # 5. Backward
        total_loss.backward()
        self.optimizer.step()

        return loss_vis.detach(), loss_aud.detach()

    def get_pareto_front(self):
        """Returns arrays compatible with previous analysis tools."""
        with torch.no_grad():
            masks = torch.sigmoid(self.mask_logits).cpu().numpy()

            # Re-eval to get final losses accurately (or store from last step)
            # Use last step's result if possible, but easier to re-run or just assume step returned it.
            # Let's re-run forward pass on CPU/NoGrad to be clean?
            # Or just use the last returned losses.
            pass
        return masks
