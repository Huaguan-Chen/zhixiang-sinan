from __future__ import annotations
import contextlib
from typing import Any, Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, groups: int = 8
    ):
        super().__init__()
        pad = k // 2
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False
            ),
            nn.GroupNorm(max(g, 1), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvNormAct(ch, ch, 3)
        g = min(8, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(max(1, g), ch)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class Encoder2d(nn.Module):
    def __init__(
        self, in_ch: int, base: int = 48, depth: int = 4, blocks_per_level: int = 2
    ):
        super().__init__()
        self.depth = depth
        chs = [base * 2**i for i in range(depth)]
        self.stem = ConvNormAct(in_ch, chs[0], 3)
        levels = []
        for i in range(depth):
            blocks = []
            if i > 0:
                blocks.append(ConvNormAct(chs[i - 1], chs[i], 3, stride=2))
            for _ in range(blocks_per_level):
                blocks.append(ResBlock2d(chs[i]))
            levels.append(nn.Sequential(*blocks))
        self.levels = nn.ModuleList(levels)
        self.out_channels = chs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for level in self.levels:
            x = level(x)
            feats.append(x)
        return feats


class TemporalMixer3d(nn.Module):
    def __init__(self, ch: int, blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        mods = []
        for _ in range(blocks):
            mods.append(
                nn.Sequential(
                    nn.Conv3d(
                        ch,
                        ch,
                        kernel_size=(3, 3, 3),
                        padding=(1, 1, 1),
                        groups=1,
                        bias=False,
                    ),
                    nn.GroupNorm(max(1, min(8, ch)), ch),
                    nn.SiLU(inplace=True),
                    nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                )
            )
        self.blocks = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.permute(0, 2, 1, 3, 4).contiguous()
        for blk in self.blocks:
            x3 = x3 + blk(x3)
        return x3.permute(0, 2, 1, 3, 4).contiguous()


class StationHistoryEncoder(nn.Module):
    def __init__(
        self,
        hist_dim: int,
        hidden: int = 128,
        out_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_proj = nn.Linear(hist_dim * 2, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        B, S, T, C = hist.shape
        x = torch.cat([hist, hist_mask], dim=-1).reshape(B * S, T, C * 2)
        x = self.in_proj(x)
        out, h = self.gru(x)
        ctx = h[-1].reshape(B, S, -1)
        return self.out(ctx)


class LeadEmbedding(nn.Module):
    def __init__(self, future_len: int, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(future_len, dim)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(T, device=device, dtype=torch.long)
        return self.emb(idx)


class RainPriorGatedUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base: int = 12,
        dropout: float = 0.0,
        lowres_size: int = 128,
        lead_chunk: int = 2,
    ):
        super().__init__()
        self.lowres_size = int(lowres_size)
        self.lead_chunk = max(1, int(lead_chunk))
        base = int(base)
        self.enc1 = nn.Sequential(ConvNormAct(in_ch, base, 3), ResBlock2d(base))
        self.down1 = ConvNormAct(base, base * 2, 3, stride=2)
        self.enc2 = ResBlock2d(base * 2)
        self.down2 = ConvNormAct(base * 2, base * 4, 3, stride=2)
        self.mid = nn.Sequential(ResBlock2d(base * 4), ResBlock2d(base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvNormAct(base * 4, base * 2, 3), ResBlock2d(base * 2)
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvNormAct(base * 2, base, 3), ResBlock2d(base))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(base, in_ch, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.up2(m)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.drop(d1)
        return self.out(d1)

    def forward(self, prior_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = prior_stack.shape
        out_dtype = prior_stack.dtype
        refined_chunks = []
        weight_chunks = []
        for t0 in range(0, T, self.lead_chunk):
            t1 = min(T, t0 + self.lead_chunk)
            x_full = prior_stack[:, t0:t1].reshape(B * (t1 - t0), C, H, W)
            if self.lowres_size > 0 and (
                H != self.lowres_size or W != self.lowres_size
            ):
                x_low = F.interpolate(
                    x_full.float(),
                    size=(self.lowres_size, self.lowres_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x_low = x_full.float()
            logits_low = self._forward_2d(x_low)
            if logits_low.shape[-2:] != (H, W):
                logits = F.interpolate(
                    logits_low, size=(H, W), mode="bilinear", align_corners=False
                )
            else:
                logits = logits_low
            logits = logits.reshape(B, t1 - t0, C, H, W)
            weights = torch.softmax(logits, dim=2).to(out_dtype)
            prior_part = prior_stack[:, t0:t1]
            refined = torch.sum(weights * prior_part, dim=2, keepdim=True)
            refined_chunks.append(refined.to(out_dtype))
            weight_chunks.append(weights)
        refined = torch.cat(refined_chunks, dim=1)
        weights = torch.cat(weight_chunks, dim=1)
        return (refined, weights)


class StationTokenAttentionHead(nn.Module):
    def __init__(
        self,
        grid_dims: Sequence[int],
        station_dim: int,
        lead_dim: int,
        prior_dim: int,
        xy_dim: int,
        out_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.05,
        ff_mult: float = 2.0,
    ):
        super().__init__()
        self.grid_dims = [int(x) for x in grid_dims]
        self.d_model = int(d_model)
        num_heads = int(num_heads)
        while self.d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = max(1, num_heads)
        self.grid_proj = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(g), nn.Linear(g, self.d_model))
                for g in self.grid_dims
            ]
        )
        self.station_proj = nn.Sequential(
            nn.LayerNorm(station_dim), nn.Linear(station_dim, self.d_model)
        )
        self.lead_proj = nn.Sequential(
            nn.LayerNorm(lead_dim), nn.Linear(lead_dim, self.d_model)
        )
        self.prior_proj = nn.Sequential(
            nn.LayerNorm(prior_dim), nn.Linear(prior_dim, self.d_model)
        )
        self.xy_proj = nn.Sequential(
            nn.LayerNorm(xy_dim), nn.Linear(xy_dim, self.d_model)
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(station_dim + lead_dim),
            nn.Linear(station_dim + lead_dim, self.d_model),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = max(self.d_model, int(round(self.d_model * float(ff_mult))))
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.out_head = nn.Linear(self.d_model, out_dim)

    def forward(
        self,
        grid_feat: torch.Tensor,
        station_ctx: torch.Tensor,
        lead_ctx: torch.Tensor,
        priors: torch.Tensor,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        B, T, S, _ = grid_feat.shape
        grid_parts = torch.split(grid_feat, self.grid_dims, dim=-1)
        if len(grid_parts) != len(self.grid_proj):
            raise RuntimeError(
                f"grid split mismatch: got {len(grid_parts)} parts, expected {len(self.grid_proj)}"
            )
        tokens = []
        for part, proj in zip(grid_parts, self.grid_proj):
            tokens.append(proj(part))
        tokens.append(self.station_proj(station_ctx))
        tokens.append(self.lead_proj(lead_ctx))
        tokens.append(self.prior_proj(priors))
        tokens.append(self.xy_proj(xy))
        kv = torch.stack(tokens, dim=-2)
        q = self.query_proj(torch.cat([station_ctx, lead_ctx], dim=-1)).unsqueeze(-2)
        n_tok = kv.shape[-2]
        q2 = q.reshape(B * T * S, 1, self.d_model)
        kv2 = kv.reshape(B * T * S, n_tok, self.d_model)
        ctx, _ = self.attn(q2, kv2, kv2, need_weights=False)
        ctx = self.norm1(q2 + self.drop(ctx))
        ctx = self.norm2(ctx + self.drop(self.ffn(ctx)))
        ctx = ctx.squeeze(1).reshape(B, T, S, self.d_model)
        return self.out_head(ctx)


class TemporalUNetStationUnionModel(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        past_ch: int,
        future_ch: int,
        hist_dim: int,
        num_targets: int,
    ):
        super().__init__()
        mcfg = cfg.get("model", {})
        base = int(mcfg.get("base_channels", 48))
        depth = int(mcfg.get("depth", 4))
        blocks_per_level = int(mcfg.get("blocks_per_level", 2))
        temporal_blocks = int(mcfg.get("temporal_blocks", 2))
        temporal_start_level = int(mcfg.get("temporal_start_level", 1))
        dropout = float(mcfg.get("dropout", 0.05))
        station_dim = int(mcfg.get("station_hidden", 128))
        lead_dim = int(mcfg.get("lead_hidden", 64))
        point_hidden = int(mcfg.get("point_hidden", 512))
        self.cfg = cfg
        self.future_len = int(cfg.get("data", {}).get("future_len", 30))
        self.station_chunk_size = int(mcfg.get("station_chunk_size", 1024))
        self.num_targets = int(num_targets)
        self.past_encoder = Encoder2d(
            past_ch, base=base, depth=depth, blocks_per_level=blocks_per_level
        )
        self.future_encoder = Encoder2d(
            future_ch, base=base, depth=depth, blocks_per_level=blocks_per_level
        )
        chs = self.future_encoder.out_channels
        self.past_proj = nn.ModuleList(
            [nn.Conv2d(chs[i], chs[i], kernel_size=1) for i in range(depth)]
        )
        self.temporal_mixers = nn.ModuleList(
            [
                TemporalMixer3d(chs[i], blocks=temporal_blocks, dropout=dropout)
                if i >= temporal_start_level
                else nn.Identity()
                for i in range(depth)
            ]
        )
        self.station_encoder = StationHistoryEncoder(
            hist_dim=hist_dim, hidden=station_dim, out_dim=station_dim, dropout=dropout
        )
        self.lead_encoder = LeadEmbedding(self.future_len, dim=lead_dim)
        self.grid_feat_dim = sum(chs)
        self.prior_channels = [
            int(x) for x in mcfg.get("prior_channels", [0, 1, 2, 3, 4, 5, 6, 7])
        ]
        prior_dim = len(self.prior_channels)
        in_dim = self.grid_feat_dim + station_dim + lead_dim + prior_dim + 2
        self.head = StationTokenAttentionHead(
            grid_dims=chs,
            station_dim=station_dim,
            lead_dim=lead_dim,
            prior_dim=prior_dim,
            xy_dim=2,
            out_dim=num_targets,
            d_model=int(mcfg.get("token_attn_dim", 128)),
            num_heads=int(mcfg.get("token_attn_heads", 4)),
            dropout=dropout,
            ff_mult=float(mcfg.get("token_attn_ff_mult", 2.0)),
        )
        self.rain_prior_kernel = int(mcfg.get("rain_prior_kernel", 7))
        if self.rain_prior_kernel < 1 or self.rain_prior_kernel % 2 == 0:
            raise ValueError(
                f"rain_prior_kernel must be a positive odd integer, got {self.rain_prior_kernel}"
            )
        self.rain_prior_patch_size = self.rain_prior_kernel * self.rain_prior_kernel
        self.rain_base_channel = int(mcfg.get("rain_base_channel", 5))
        self.rain_refine_channels = [
            int(x) for x in mcfg.get("rain_refine_channels", [3, 4, 5])
        ]
        if len(self.rain_refine_channels) != 3:
            raise ValueError(
                f"rain_refine_channels should contain 3 channels, got {self.rain_refine_channels}"
            )
        self.rain_prior_refiner = RainPriorGatedUNet(
            in_ch=len(self.rain_refine_channels),
            base=int(mcfg.get("rain_refiner_base", 12)),
            dropout=float(mcfg.get("rain_refiner_dropout", 0.0)),
            lowres_size=int(mcfg.get("rain_refiner_lowres_size", 128)),
            lead_chunk=int(mcfg.get("rain_refiner_lead_chunk", 2)),
        )
        self.rain_prior_attn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, point_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(point_hidden // 2, self.rain_prior_patch_size),
        )
        nn.init.zeros_(self.rain_prior_attn[-1].weight)
        nn.init.zeros_(self.rain_prior_attn[-1].bias)
        final = getattr(self.head, "out_head", None)
        if isinstance(final, nn.Linear) and final.out_features >= 1:
            with torch.no_grad():
                final.weight[0].zero_()
                final.bias[0].zero_()
        self.rain_residual_scale = float(mcfg.get("rain_residual_scale", 0.15))
        self.rain_residual_limit = float(mcfg.get("rain_residual_limit", 0.08))
        self.rain_softplus_beta = float(mcfg.get("rain_softplus_beta", 200.0))
        self.nonnegative_output_indices = [
            int(x) for x in mcfg.get("nonnegative_output_indices", [1, 13, 14])
        ]
        self.nonnegative_softplus_beta = float(
            mcfg.get("nonnegative_softplus_beta", 200.0)
        )

    def _encode_past_context(self, past_grid: torch.Tensor) -> List[torch.Tensor]:
        B, Tp, C, H, W = past_grid.shape
        x = past_grid.reshape(B * Tp, C, H, W)
        feats = self.past_encoder(x)
        ctx = []
        for i, f in enumerate(feats):
            _, ch, h, w = f.shape
            f = f.reshape(B, Tp, ch, h, w).mean(dim=1)
            ctx.append(self.past_proj[i](f))
        return ctx

    def _encode_future(
        self, future_grid: torch.Tensor, past_ctx: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        B, T, C, H, W = future_grid.shape
        x = future_grid.reshape(B * T, C, H, W)
        feats = self.future_encoder(x)
        outs = []
        for i, f in enumerate(feats):
            _, ch, h, w = f.shape
            f = f.reshape(B, T, ch, h, w)
            pc = past_ctx[i].unsqueeze(1)
            f = f + pc
            f = self.temporal_mixers[i](f)
            outs.append(f)
        return outs

    def _sample_feat_level(
        self, feat: torch.Tensor, station_xy: torch.Tensor, s0: int, s1: int
    ) -> torch.Tensor:
        B, T, C, h, w = feat.shape
        out_dtype = feat.dtype
        xy = station_xy[:, s0:s1, :]
        S = xy.shape[1]
        fmap = feat.reshape(B * T, C, h, w)
        grid = xy[:, None, :, None, :].expand(B, T, S, 1, 2).reshape(B * T, S, 1, 2)
        ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if fmap.is_cuda
            else contextlib.nullcontext()
        )
        with ctx:
            val = F.grid_sample(
                fmap.float(),
                grid.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        val = val.squeeze(-1).permute(0, 2, 1).reshape(B, T, S, C)
        return val.to(out_dtype)

    def _sample_future_priors(
        self, future_grid: torch.Tensor, station_xy: torch.Tensor, s0: int, s1: int
    ) -> torch.Tensor:
        B, T, C, H, W = future_grid.shape
        out_dtype = future_grid.dtype
        x = future_grid[:, :, self.prior_channels, :, :]
        Cp = x.shape[2]
        xy = station_xy[:, s0:s1, :]
        S = xy.shape[1]
        fmap = x.reshape(B * T, Cp, H, W)
        grid = xy[:, None, :, None, :].expand(B, T, S, 1, 2).reshape(B * T, S, 1, 2)
        ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if fmap.is_cuda
            else contextlib.nullcontext()
        )
        with ctx:
            val = F.grid_sample(
                fmap.float(),
                grid.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        val = val.squeeze(-1).permute(0, 2, 1).reshape(B, T, S, Cp)
        return val.to(out_dtype)

    def _make_refined_rain_prior_map(self, future_grid: torch.Tensor) -> torch.Tensor:
        prior_stack = future_grid[:, :, self.rain_refine_channels, :, :]
        refined_prior, _ = self.rain_prior_refiner(prior_stack)
        return refined_prior.to(future_grid.dtype)

    def _sample_prior_map_patch(
        self, prior_map: torch.Tensor, station_xy: torch.Tensor, s0: int, s1: int
    ) -> torch.Tensor:
        B, T, C, H, W = prior_map.shape
        if C != 1:
            raise ValueError(f"prior_map should have one channel, got {C}")
        out_dtype = prior_map.dtype
        k = int(self.rain_prior_kernel)
        r = k // 2
        x = prior_map.reshape(B * T, 1, H, W)
        xy = station_xy[:, s0:s1, :]
        S = xy.shape[1]
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=prior_map.device, dtype=torch.float32),
            torch.arange(-r, r + 1, device=prior_map.device, dtype=torch.float32),
            indexing="ij",
        )
        ox = xx.reshape(-1) * (2.0 / max(W - 1, 1))
        oy = yy.reshape(-1) * (2.0 / max(H - 1, 1))
        offsets = torch.stack([ox, oy], dim=-1)
        base = xy[:, None, :, None, :].expand(B, T, S, 1, 2)
        grid = base + offsets[None, None, None, :, :]
        grid = torch.clamp(grid, -1.0, 1.0)
        grid = grid.reshape(B * T, S * k * k, 1, 2)
        ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with ctx:
            val = F.grid_sample(
                x.float(),
                grid.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        val = val.squeeze(1).squeeze(-1).reshape(B, T, S, k * k)
        return val.to(out_dtype)

    def _sample_zrmax_prior_patch(
        self, future_grid: torch.Tensor, station_xy: torch.Tensor, s0: int, s1: int
    ) -> torch.Tensor:
        B, T, C, H, W = future_grid.shape
        k = int(self.rain_prior_kernel)
        r = k // 2
        ch = int(self.rain_base_channel)
        x = future_grid[:, :, ch : ch + 1, :, :].reshape(B * T, 1, H, W)
        xy = station_xy[:, s0:s1, :]
        Sc = xy.shape[1]
        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ox = 2.0 * float(dx) / max(float(W - 1), 1.0)
                oy = 2.0 * float(dy) / max(float(H - 1), 1.0)
                offsets.append([ox, oy])
        off = torch.tensor(offsets, dtype=xy.dtype, device=xy.device)
        patch_xy = xy[:, :, None, :] + off[None, None, :, :]
        patch_xy = patch_xy.clamp(-1.0, 1.0)
        grid = (
            patch_xy[:, None, :, :, :]
            .expand(B, T, Sc, k * k, 2)
            .reshape(B * T, Sc * k * k, 1, 2)
        )
        ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with ctx:
            val = F.grid_sample(
                x.float(),
                grid.float(),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        patch = val.squeeze(1).squeeze(-1).reshape(B, T, Sc, k * k)
        return patch.to(future_grid.dtype)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past_grid = batch["past_grid"]
        future_grid = batch["future_grid"]
        station_xy = batch["station_xy"]
        station_hist = batch["station_hist"]
        hist_mask = batch["station_hist_mask"]
        B, T, _, _, _ = future_grid.shape
        S = station_xy.shape[1]
        past_ctx = self._encode_past_context(past_grid)
        future_feats = self._encode_future(future_grid, past_ctx)
        refined_rain_prior = self._make_refined_rain_prior_map(future_grid)
        station_ctx = self.station_encoder(station_hist, hist_mask)
        lead_ctx = self.lead_encoder(T, future_grid.device)
        preds = []
        chunk = max(1, int(self.station_chunk_size))
        for s0 in range(0, S, chunk):
            s1 = min(S, s0 + chunk)
            sampled = [
                self._sample_feat_level(f, station_xy, s0, s1) for f in future_feats
            ]
            grid_feat = torch.cat(sampled, dim=-1)
            priors = self._sample_future_priors(future_grid, station_xy, s0, s1)
            sc = station_ctx[:, s0:s1, :].unsqueeze(1).expand(-1, T, -1, -1)
            le = lead_ctx[None, :, None, :].expand(B, -1, s1 - s0, -1)
            xy = station_xy[:, s0:s1, :].unsqueeze(1).expand(-1, T, -1, -1)
            head_dtype = grid_feat.dtype
            sc = sc.to(head_dtype)
            le = le.to(head_dtype)
            priors = priors.to(head_dtype)
            xy = xy.to(head_dtype)
            inp = torch.cat([grid_feat, sc, le, priors, xy], dim=-1)
            out = self.head(grid_feat, sc, le, priors, xy)
            prior_patch = self._sample_prior_map_patch(
                prior_map=refined_rain_prior, station_xy=station_xy, s0=s0, s1=s1
            )
            attn_logits = self.rain_prior_attn(inp).float()
            attn = torch.softmax(attn_logits, dim=-1).to(prior_patch.dtype)
            local_refined_prior_norm = torch.sum(attn * prior_patch, dim=-1).to(
                out.dtype
            )
            raw_pre6_residual = out[..., 0]
            bounded_residual = self.rain_residual_limit * torch.tanh(raw_pre6_residual)
            out_pre_raw = local_refined_prior_norm + bounded_residual
            out_pre = torch.clamp(out_pre_raw, min=0.0)
            out = torch.cat([out_pre.unsqueeze(-1), out[..., 1:]], dim=-1)
            if self.nonnegative_output_indices:
                valid_idx = [
                    int(i)
                    for i in self.nonnegative_output_indices
                    if 0 <= int(i) < out.shape[-1]
                ]
                if valid_idx:
                    idx = torch.tensor(valid_idx, device=out.device, dtype=torch.long)
                    mask_nonneg = torch.zeros(
                        out.shape[-1], device=out.device, dtype=torch.bool
                    )
                    mask_nonneg[idx] = True
                    view_shape = [1] * (out.ndim - 1) + [out.shape[-1]]
                    out_soft = F.softplus(out, beta=self.nonnegative_softplus_beta)
                    out = torch.where(mask_nonneg.view(*view_shape), out_soft, out)
            preds.append(out)
        pred = (
            torch.cat(preds, dim=2)
            if preds
            else torch.empty(B, T, 0, self.num_targets, device=future_grid.device)
        )
        result = {"preds": pred}
        return result
