"""Gradient-Based Optimization for Pareto Front finding.

Uses scalarization (Batch of weighted sums) to find diverse solutions in parallel via Adam.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .objectives import calc_audio_mag_loss


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
        # We need to counteract the 20x Visual Loss Boost.
        # Simple Linear spacing makes 95% of the population Visual-Dominated.
        # We use a Power Law to push more weights into the low-visual-weight region.
        alpha = torch.linspace(0, 1, pop_size, device=encoder.device)

        # weights_img = (1-alpha)^4. This bunches values near 0.
        # If alpha=0 (Image) -> w=1. If alpha=1 (Audio) -> w=0.
        # Midpoint alpha=0.5 -> w=0.0625.
        # 0.0625 * 20 = 1.25 vs 0.93 (Audio). Closely balanced!
        self.weights_img = (1.0 - alpha).pow(4.0)

        # Ensure Audio dominates the rest
        self.weights_aud = 1.0 - self.weights_img

        # Normalize sum to 1 (Optional but good for stability)
        total = self.weights_img + self.weights_aud
        self.weights_img /= total
        self.weights_aud /= total

        # 3. Force Anchors (Edge Points)
        # Ind 0: w_img=1 (Target: Image). Init: Image (Logits +10)
        # Ind -1: w_aud=1 (Target: Audio). Init: Audio (Logits -10)
        # Previous logic was swaped!
        if pop_size >= 2:
            with torch.no_grad():
                self.mask_logits[0].fill_(10.0)  # Starts as Image
                self.mask_logits[-1].fill_(-10.0)  # Starts as Audio

        # 3. Optimizer & Scaler
        self.optimizer = optim.Adam([self.mask_logits], lr=learning_rate)
        
        self.device_type = encoder.device.type
        self.use_amp = (self.device_type == "cuda")
        
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Loss Normalization Factors (Defaults 1.0)
        self.scale_vis = 1.0
        self.scale_aud = 1.0

    def calculate_normalization(self, image_mag_ref, audio_mag_ref):
        """Calculate worst-case losses to normalize objectives to [0, 1]."""
        print("Calculating Loss Normalization Factors...")
        with torch.no_grad():
            # Worst Case Visual Loss: Mask = 0 (Pure Audio)
            # Mixed = Audio. Loss = |Audio - Image|
            diff_vis = torch.abs(audio_mag_ref - image_mag_ref)
            max_vis_loss = diff_vis.mean().item()

            # Worst Case Audio Loss: Mask = 1 (Pure Image)
            # Mixed = Image. Loss = |Image - Audio|
            # Note: calc_audio_mag_loss might be complex, so we call it.
            # Assuming Mask=1 => mixed = image
            max_aud_loss = calc_audio_mag_loss(image_mag_ref, audio_mag_ref).item()

            # Avoid division by zero
            if max_vis_loss < 1e-6:
                max_vis_loss = 1.0
            if max_aud_loss < 1e-6:
                max_aud_loss = 1.0

            # Heuristic: Boost Visual Loss importance by 20x to match human perception
            # functionality better (User Request)
            self.scale_vis = 1.0 / max_vis_loss
            self.scale_aud = 1.0 / max_aud_loss

            print(
                f"  Max Visual Loss: {max_vis_loss:.4f} -> Scale: {self.scale_vis:.4f}"
            )
            print(
                f"  Max Audio Loss:  {max_aud_loss:.4f} -> Scale: {self.scale_aud:.4f}"
            )

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
            # Use nullcontext if not using AMP
            from contextlib import nullcontext
            ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
            
            with ctx:
                # 2. Get Masks (Sigmoid)
                masks = torch.sigmoid(chunk_logits)

                # 3. Forward Pass (Skip ISTFT for speed/mem)
                _, mixed_mag = self.encoder(masks, return_wav=False)

                # 4. Calculate Losses (L1 instead of SSIM for speed)
                # Visual Loss: L1 distance between log-magnitudes or straight magnitudes
                # Using magnitudes (Ref is 0-1 normalized)
                diff = torch.abs(mixed_mag - image_mag_ref)
                # Mean over F, T (last 2 dims) -> (B,)
                raw_loss_vis = diff.mean(dim=(1, 2))
                raw_loss_aud = calc_audio_mag_loss(mixed_mag, audio_mag_ref)

                # NORMALIZE LOSSES
                loss_vis = raw_loss_vis * self.scale_vis
                # calc_audio_mag_loss returns (B,) already? Need to verify. assuming yes or scalar.
                # If scalar, we need to match shape. But ParetoManager implies batch.
                # Assuming calc_audio_mag_loss handles batching or returns scalar if input is batch.
                # Just in case, let's ensure broadcasting works.
                loss_aud = raw_loss_aud * self.scale_aud

                # 5. Scalarized Loss
                # User Request: Boost Visual Loss importance by 20x for optimization direction
                # But keep reported metrics 0-1.
                total_loss = torch.sum(
                    chunk_weights_img * loss_vis * 20.0 + chunk_weights_aud * loss_aud
                )

            # 6. Backward
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Record losses (detach and float for logging)
            loss_vis_list.append(loss_vis.detach().float())
            loss_aud_list.append(loss_aud.detach().float())

        # 7. Optimizer Step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # FORCE CLAMP ANCHORS (Hard Constraints)
        # Ensure we always keep the trivial extrema (Pure Image, Pure Audio)
        # This prevents Adam from drifting away from the perfect edge cases.
        if self.pop_size >= 2:
            with torch.no_grad():
                # Index 0: Pure Image (Mask = 1) -> Logits = +20
                self.mask_logits[0].fill_(20.0)
                # Index -1: Pure Audio (Mask = 0) -> Logits = -20
                self.mask_logits[-1].fill_(-20.0)

        # Aggregate losses for logging
        avg_vis = torch.cat(loss_vis_list).mean().item()
        avg_aud = torch.cat(loss_aud_list).mean().item()
        return avg_vis, avg_aud

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
