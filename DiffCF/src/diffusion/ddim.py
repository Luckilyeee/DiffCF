import torch


def get_ddim_timesteps(ddim_steps, ddpm_steps, start_ratio=1.0):
    start_step = int(ddpm_steps * start_ratio)
    start_step = min(max(start_step, 1), ddpm_steps)
    steps = torch.linspace(start_step - 1, 0, ddim_steps, dtype=torch.long)
    return steps


def predict_x0_from_eps(x_t, eps, alpha_bar_t):
    return (x_t - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()


def ddim_step(x_t, eps, t, t_prev, alpha_bar, eta=0.0):
    alpha_bar_t = alpha_bar[t]
    alpha_bar_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
    x0 = predict_x0_from_eps(x_t, eps, alpha_bar_t)
    sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
    noise = torch.randn_like(x_t)
    dir_term = (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps
    x_prev = alpha_bar_prev.sqrt() * x0 + dir_term + sigma * noise
    return x_prev, x0

