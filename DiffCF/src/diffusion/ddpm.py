import torch
import torch.nn.functional as F
from .schedules import cosine_beta_schedule, linear_beta_schedule


class GaussianDiffusion:
    def __init__(self, timesteps=1000, schedule="cosine"):
        self.timesteps = timesteps
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bar.to(x0.device)
        sqrt_alpha_bar = alpha_bar[t].view(-1, 1, 1).sqrt()
        sqrt_one_minus = (1 - alpha_bar[t]).view(-1, 1, 1).sqrt()
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def training_losses(self, model, x0, t, loss_type="mse", lambda_tv=0.01, lambda_smooth=0.0):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        pred_noise = model(x_t, t)


        loss_mse = F.mse_loss(pred_noise, noise)
        if loss_type == "mse":
            return loss_mse


        alpha_bar_t = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_t = (1 - alpha_bar_t).sqrt()

        x0_hat = (x_t - sqrt_one_minus_t * pred_noise) / sqrt_alpha_bar_t


        diff1 = x0_hat[:, :, 1:] - x0_hat[:, :, :-1]
        loss_tv = torch.mean(torch.abs(diff1))


        loss = loss_mse + lambda_tv * loss_tv

        return loss
