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

        # Force Anchors (Edge Points)
        # Ind 0: Weight Img=0, Aud=1 -> Wants Pure Audio (Mask=0, Logits=-10)
        # Ind -1: Weight Img=1, Aud=0 -> Wants Pure Image (Mask=1, Logits=+10)
        if pop_size >= 2:
            with torch.no_grad():
                self.mask_logits[0].fill_(-10.0)
                self.mask_logits[-1].fill_(10.0)

        # 2. Assign Scalarization Weights
        # Linearly space weights from [1, 0] (Pure Image) to [0, 1] (Pure Audio)
        # This forces the population to spread across the front.
        alpha = torch.linspace(0, 1, pop_size, device=encoder.device)
        # w_img goes 1 -> 0
        # w_aud goes 0 -> 1
        self.weights_img = 1.0 - alpha
        self.weights_aud = alpha

        # 3. Optimizer & Scaler
        self.optimizer = optim.Adam([self.mask_logits], lr=learning_rate)
        self.scaler = torch.amp.GradScaler("cuda")

    def optimize_step(self, image_mag_ref, audio_mag_ref, micro_batch_size=5):
        """Performs one step of gradient descent using micro-batching and AMP."""
        self.optimizer.zero_grad()

        # micro_batch_size is now passed from caller
        total_pop = self.pop_size

        loss_vis_list = []
        loss_aud_list = []

        for i in range(0, total_pop, micro_batch_size):
            chunk_logits = self.mask_logits[i : i + micro_batch_size]
            chunk_weights_img = self.weights_img[i : i + micro_batch_size]
            chunk_weights_aud = self.weights_aud[i : i + micro_batch_size]

            # AMP Context
            with torch.amp.autocast("cuda"):
                # 2. Get Masks (Sigmoid)
                masks = torch.sigmoid(chunk_logits)

                # 3. Forward Pass (Skip ISTFT for speed/mem)
                _, mixed_mag = self.encoder(masks, return_wav=False)

                # 4. Calculate Losses (L1 instead of SSIM for speed)
                # Visual Loss: L1 distance between log-magnitudes or straight magnitudes
                # Using magnitudes (Ref is 0-1 normalized)
                # We need to average over pixels to get a scalar per batch item
                # F.l1_loss returns scalar by default (mean), but we need (B,)
                # So we calculate abs diff and mean manually or use reduction='none'

                diff = torch.abs(mixed_mag - image_mag_ref)
                # Mean over F, T (last 2 dims) -> (B,)
                loss_vis = diff.mean(dim=(1, 2))

                loss_aud = calc_audio_mag_loss(mixed_mag, audio_mag_ref)

                # 5. Scalarized Loss
                total_loss = torch.sum(
                    chunk_weights_img * loss_vis + chunk_weights_aud * loss_aud
                )

            # 6. Backward with Scaler
            self.scaler.scale(total_loss).backward()

            # Record losses (detach and float for logging)
            loss_vis_list.append(loss_vis.detach().float())
            loss_aud_list.append(loss_aud.detach().float())

            # Cleanup graph for this chunk
            del masks, mixed_mag, total_loss, diff, chunk_logits

        # Optimizer Step with Scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()

        final_vis = torch.cat(loss_vis_list)
        final_aud = torch.cat(loss_aud_list)

        return final_vis, final_aud

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
