import torch
import torch.nn.functional as F
from .temporal_cone import TemporalConeProjector


def _normalize_grad(g, eps=1e-6):
    norm = torch.sqrt(torch.sum(g ** 2, dim=(1, 2), keepdim=True))
    return g / (norm + eps)


def _gaussian_kernel1d(sigma, device):
    radius = int(3 * sigma)
    xs = torch.arange(-radius, radius + 1, device=device).float()
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def _smooth_grad(g, sigma):
    if sigma <= 0:
        return g
    kernel = _gaussian_kernel1d(sigma, g.device)
    g_padded = F.pad(g, (kernel.shape[-1] // 2,) * 2, mode="reflect")
    return F.conv1d(g_padded, kernel.expand(g.shape[1], 1, -1), groups=g.shape[1])


def _augmentations(x, rng, max_shift=3, scale_std=0.01, noise_std=0.01):
    shift = int(rng.randint(-max_shift, max_shift + 1))
    scale = 1.0 + float(rng.randn() * scale_std)
    noise = torch.randn_like(x) * noise_std
    return torch.roll(x * scale + noise, shifts=shift, dims=-1)


def compute_guidance(x_t, t, eps_pred, diffusion, classifier, x_orig, target, cfg):
    alpha_bar = diffusion.alpha_bar.to(x_t.device)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1)
    x0_hat = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()

    x0_hat = x0_hat.clamp(-3, 3)

    x0_hat.requires_grad_(True)
    logp = None

    if cfg["stabilization"]["mode"] == "aug_avg":
        rng = torch.Generator(device=x_t.device)
        rng.manual_seed(0)
        k = cfg["stabilization"]["aug_k"]
        logps = []
        for _ in range(k):
            x_aug = _augmentations(x0_hat, rng)
            logits = classifier(x_aug)
            logps.append(F.log_softmax(logits, dim=-1)[:, target])
        logp = torch.stack(logps, dim=0).mean(dim=0)
    else:
        logits = classifier(x0_hat)
        logp = F.log_softmax(logits, dim=-1)[:, target]

    g_cls = torch.autograd.grad(logp.sum(), x0_hat, retain_graph=True)[0]
    if cfg["stabilization"]["mode"] == "grad_smooth":
        g_cls = _smooth_grad(g_cls, cfg["stabilization"]["grad_smooth_sigma"])


    dist = torch.abs(x0_hat - x_orig).mean(dim=(1, 2))
    g_dist = torch.autograd.grad(dist.sum(), x0_hat, retain_graph=True)[0]

    g_total = cfg["sampling"]["w_cls"] * _normalize_grad(g_cls) - cfg["sampling"]["w_dist"] * _normalize_grad(g_dist)

    w_smooth = cfg["sampling"]["w_smooth"]
    if w_smooth > 0:
        d2 = x0_hat[:, :, 2:] - 2 * x0_hat[:, :, 1:-1] + x0_hat[:, :, :-2]
        smooth = (d2 ** 2).mean(dim=(1, 2))
        g_smooth = torch.autograd.grad(smooth.sum(), x0_hat, retain_graph=True)[0]
        g_total = g_total - w_smooth * _normalize_grad(g_smooth)

    return x0_hat.detach(), g_total.detach()
