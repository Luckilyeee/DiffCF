
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # 确保导入了 F


def sinusoidal_time_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _group_norm_groups(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='replicate')
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='replicate')
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels, base_channels=64, depth=3, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = base_channels
        for _ in range(depth):
            self.downs.append(nn.ModuleList([
                ResBlock(ch, ch * 2, time_dim),
                nn.Conv1d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),
            ]))
            ch *= 2

        self.mid = ResBlock(ch, ch, time_dim)

        for _ in range(depth):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                ResBlock(ch + (ch // 2), ch // 2, time_dim),
            ]))
            ch //= 2

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.act = nn.SiLU()

    def forward(self, x, t):
        assert x.ndim == 3, "Expected [B, C, T]"
        orig_len = x.shape[-1]


        divisor = 2 ** len(self.downs)
        if orig_len % divisor != 0:
            pad_len = divisor - (orig_len % divisor)

            x = F.pad(x, (0, pad_len), mode='replicate')

        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = self.in_conv(x)
        skips = []
        for block, down in self.downs:
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, t_emb)
        for up, block in self.ups:
            h = up(h)
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                min_len = min(h.shape[-1], skip.shape[-1])
                h = h[..., :min_len]
                skip = skip[..., :min_len]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        h = self.act(self.out_norm(h))
        h = self.out_conv(h)

        return h[..., :orig_len]
