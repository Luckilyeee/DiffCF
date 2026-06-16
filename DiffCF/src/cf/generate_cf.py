import torch
from ..diffusion.ddim import get_ddim_timesteps, ddim_step
from .guidance import compute_guidance


def _pick_target_rule(probs, rule=None):

    if rule in (None, "next_best"):
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).indices
        if top2.shape[-1] == 1:
            return top2[:, 0]
        return top2[:, 1]
    return torch.full((probs.shape[0],), int(rule), device=probs.device, dtype=torch.long)


def _clip_grad(g, max_norm):
    if max_norm is None:
        return g
    max_norm = float(max_norm)
    if max_norm <= 0:
        return g
    norm = torch.sqrt(torch.sum(g ** 2, dim=(1, 2), keepdim=True))
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return g * scale


def _step_size_for_t(step_size, t, timesteps, schedule="constant", min_ratio=0.2):
    if schedule == "linear":
        denom = max(timesteps - 1, 1)
        ratio = 1.0 - float(t) / float(denom)
        ratio = max(min_ratio, ratio)
        return step_size * ratio
    return step_size


def _should_apply_guidance(t, timesteps, start_ratio):
    if start_ratio is None:
        return True
    start_ratio = float(start_ratio)
    if start_ratio >= 1.0:
        return True
    start_t = int(timesteps * start_ratio)
    return int(t) <= start_t


def generate_counterfactual(model, diffusion, classifier, x_orig, cfg, target=None):
    device = x_orig.device
    classifier.eval()
    model.eval()

    with torch.no_grad():
        probs = classifier.predict_proba(x_orig)
    if target is None:
        if probs.shape[-1] == 2:
            pred = probs.argmax(dim=-1)
            target = 1 - pred
        else:
            rule = cfg.get("sampling", {}).get("target_rule")
            target = _pick_target_rule(probs, rule)

    sampling_cfg = cfg.get("sampling", {})
    max_retries = int(sampling_cfg.get("max_retries", 3))
    start_ratio = float(sampling_cfg.get("start_ratio", 0.6))
    retry_start_ratio_inc = float(sampling_cfg.get("retry_start_ratio_inc", 0.1))
    w_cls_base = float(sampling_cfg.get("w_cls", 2.0))
    retry_w_cls_mult = float(sampling_cfg.get("retry_w_cls_mult", 1.5))
    step_size = float(sampling_cfg.get("step_size", 0.1))
    eta = float(sampling_cfg.get("eta", 0.0))
    guidance_start_ratio = sampling_cfg.get("guidance_start_ratio", 1.0)
    grad_clip_norm = sampling_cfg.get("grad_clip_norm")
    step_size_schedule = sampling_cfg.get("step_size_schedule", "constant")
    step_size_min_ratio = float(sampling_cfg.get("step_size_min_ratio", 0.2))

    best_cf = None
    for retry in range(max_retries):
        start_ratio_retry = min(1.0, start_ratio + retry * retry_start_ratio_inc)
        w_cls = w_cls_base * (retry_w_cls_mult ** retry)
        cfg_local = cfg.copy()
        cfg_local["sampling"] = dict(sampling_cfg, w_cls=w_cls, start_ratio=start_ratio_retry)

        timesteps = get_ddim_timesteps(cfg["diffusion"]["ddim_steps"], diffusion.timesteps, start_ratio=start_ratio_retry)
        t_start = timesteps[0].item()
        t_batch = torch.full((x_orig.shape[0],), t_start, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x_orig, t_batch)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((x_orig.shape[0],), t.item(), device=device, dtype=torch.long)
            apply_guidance = _should_apply_guidance(t.item(), diffusion.timesteps, guidance_start_ratio)
            if apply_guidance:
                with torch.enable_grad():
                    x_t = x_t.detach().requires_grad_(True)
                    eps_pred = model(x_t, t_batch)
                    x0_hat, g_total = compute_guidance(
                        x_t,
                        t_batch,
                        eps_pred,
                        diffusion,
                        classifier,
                        x_orig,
                        target,
                        cfg_local,
                    )
                    g_total = _clip_grad(g_total, grad_clip_norm)
                    step_size_t = _step_size_for_t(step_size, t.item(), diffusion.timesteps,
                                                   step_size_schedule, step_size_min_ratio)
                    x_t = (x_t + step_size_t * g_total).detach()
            else:
                x_t = x_t.detach()
            t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1
            eps_pred = model(x_t, t_batch)
            x_t, x0_hat = ddim_step(x_t, eps_pred, t.item(), t_prev, diffusion.alpha_bar.to(device), eta=eta)

        x_cf = x0_hat.detach()
        with torch.no_grad():
            cf_pred = classifier.predict_proba(x_cf).argmax(dim=-1)
        if torch.all(cf_pred == target):
            best_cf = x_cf
            break
        best_cf = x_cf

    return best_cf, target
