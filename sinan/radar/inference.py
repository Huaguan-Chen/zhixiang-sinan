import os
import json
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .base import MutiPhyPreNET3D
from .diffusion_new import get_model as get_model_new
from .diffusion_zero import get_model as get_model_0

TS_FMT = "%Y%m%d_%H%M%S"


class JSONForecastDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_file: str,
        dt_minutes: int = 6,
        anchor_time: datetime = datetime(2023, 1, 1, 0, 0, 0),
        normalize_div: float = 800.0,
        crop_last_dim_to: Optional[int] = 512,
    ):
        self.root_dir = root_dir
        self.json_file = json_file
        self.dt_minutes = int(dt_minutes)
        self.anchor_time = anchor_time
        self.normalize_div = float(normalize_div)
        self.crop_last_dim_to = crop_last_dim_to
        path = os.path.join(root_dir, json_file)
        with open(path, "r") as f:
            self.windows = json.load(f)
        if len(self.windows) == 0:
            raise RuntimeError(f"Empty json file: {path}")
        first = self.windows[0]
        self.T_in = len(first["past"])
        self.T_out = len(first["future"])

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _get_start_str(window: dict) -> str:
        if "start" in window and window["start"] is not None:
            return window["start"]
        fname = os.path.basename(window["past"][0]["data"])
        return fname.replace("_data.npy", "")

    def _compute_frame_index(self, start_str: str) -> int:
        t = datetime.strptime(start_str, TS_FMT)
        delta_sec = (t - self.anchor_time).total_seconds()
        return int(delta_sec // (self.dt_minutes * 60))

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x_data, x_mask = ([], [])
        for rec in w["past"]:
            d = np.load(os.path.join(self.root_dir, rec["data"])).astype(np.float32)
            m = np.load(os.path.join(self.root_dir, rec["mask"])).astype(np.float32)
            if self.crop_last_dim_to is not None:
                d = d[..., : self.crop_last_dim_to]
                m = m[..., : self.crop_last_dim_to]
            x_data.append(d)
            x_mask.append(m)
        x_data = np.stack(x_data, axis=0) / self.normalize_div
        x_mask = np.stack(x_mask, axis=0)
        start_str = self._get_start_str(w)
        frame_idx = self._compute_frame_index(start_str)
        future_ts = []
        for rec in w["future"]:
            fname = os.path.basename(rec["data"])
            future_ts.append(fname.replace("_data.npy", ""))
        return (
            torch.from_numpy(x_data).float(),
            torch.from_numpy(x_mask).float(),
            torch.tensor(frame_idx, dtype=torch.long),
            future_ts,
            start_str,
        )


def collate_batch(batch):
    x_data = torch.stack([b[0] for b in batch], dim=0)
    x_mask = torch.stack([b[1] for b in batch], dim=0)
    frame_idx = torch.stack([b[2] for b in batch], dim=0)
    future_ts = [b[3] for b in batch]
    window_start = [b[4] for b in batch]
    return (x_data, x_mask, frame_idx, future_ts, window_start)


def strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_base_model(ckpt_path: str, device: torch.device):
    model = MutiPhyPreNET3D().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(strip_module_prefix(state), strict=True)
    return model.eval()


def load_diffusion_models(diff_new_ckpt: str, diff_0_ckpt: str, device: torch.device):
    models = []
    for factory, checkpoint in [
        (get_model_new, diff_new_ckpt),
        (get_model_0, diff_0_ckpt),
    ]:
        model = factory().to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model" in state:
            state = state["model"]
        state = strip_module_prefix(state)
        state.pop("loss_weight", None)
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
    return tuple(models)


def make_y_mask_from_x_mask(x_mask: torch.Tensor, t_out: int) -> torch.Tensor:
    m = x_mask.float().mean(dim=1, keepdim=True)
    if m.dim() == 6:
        m = m.repeat(1, t_out, 1, 1, 1, 1)
    elif m.dim() == 5:
        m = m.repeat(1, t_out, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported x_mask dim: {m.dim()}")
    return m


def remove_small_connected_regions(x: torch.Tensor, min_size: int = 4) -> torch.Tensor:
    try:
        from scipy import ndimage
    except Exception as e:
        raise ImportError("需要 scipy 才能做连通域过滤：pip install scipy") from e
    x_cpu = x.detach().float().cpu().numpy()
    if x_cpu.ndim != 4:
        raise ValueError(f"pred_y 期望形状 (B,T,H,W)，实际 {x_cpu.shape}")
    B, T, _, _ = x_cpu.shape
    out = x_cpu.copy()
    for b in range(B):
        for t in range(T):
            img = out[b, t]
            fg = img > 0
            if not fg.any():
                continue
            lab, num = ndimage.label(fg)
            if num == 0:
                continue
            sizes = ndimage.sum(fg, lab, index=np.arange(1, num + 1))
            for i, sz in enumerate(sizes, start=1):
                if sz < min_size:
                    img[lab == i] = 0.0
            out[b, t] = img
    return torch.from_numpy(out).to(x.device)


def run_two_diffusions(
    diffusion_0,
    diffusion_new,
    x_zmax_norm: torch.Tensor,
    x_pre_zmax_norm: torch.Tensor,
    y_mask_zmax: torch.Tensor,
    diff_threshold: float,
    diff_min_size: int,
):
    pred_y_0 = diffusion_0.sample(x_zmax_norm, x_pre_zmax_norm)
    pred_y_0 = (
        torch.where(pred_y_0 < diff_threshold, torch.zeros_like(pred_y_0), pred_y_0)
        * y_mask_zmax
    )
    pred_y_0 = remove_small_connected_regions(pred_y_0, min_size=diff_min_size)
    pred_y_new = diffusion_new.sample(x_zmax_norm, x_pre_zmax_norm)
    pred_y_new = (
        torch.where(
            pred_y_new < diff_threshold, torch.zeros_like(pred_y_new), pred_y_new
        )
        * y_mask_zmax
    )
    pred_y_new = remove_small_connected_regions(pred_y_new, min_size=diff_min_size)
    return (pred_y_0, pred_y_new)


def save_one_array(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = arr.astype(np.float16, copy=False)
    npz_path = os.path.splitext(path)[0] + ".npz"
    np.savez_compressed(npz_path, arr=arr)


def maybe_denorm(arr: np.ndarray, enabled: bool, scale: float) -> np.ndarray:
    return arr * scale if enabled else arr


@torch.no_grad()
def infer_radar(
    root_dir: str,
    json_file: str,
    save_root: str,
    ckpt_path: str,
    diff_new_ckpt: str,
    diff_0_ckpt: str,
    batch_size: int = 4,
    num_workers: int = 4,
    dt_minutes: int = 6,
    anchor_time: datetime = datetime(2023, 1, 1, 0, 0, 0),
    lat_range: Tuple[float, float] = (37.34, 42.46),
    lon_range: Tuple[float, float] = (113.96, 119.08),
    normalize_div: float = 800.0,
    crop_last_dim_to: Optional[int] = 512,
    diff_threshold: float = 0.1,
    diff_min_size: int = 4,
    save_denorm_out_all: bool = False,
    save_denorm_pred2d: bool = False,
    device_name: str = "auto",
):
    rank, world_size = (0, 1)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else device_name
    )
    dataset = JSONForecastDataset(
        root_dir=root_dir,
        json_file=json_file,
        dt_minutes=dt_minutes,
        anchor_time=anchor_time,
        normalize_div=normalize_div,
        crop_last_dim_to=crop_last_dim_to,
    )
    sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_batch,
    )
    model = load_base_model(ckpt_path, device)
    diffusion_new, diffusion_0 = load_diffusion_models(
        diff_new_ckpt, diff_0_ckpt, device
    )
    json_tag = os.path.splitext(os.path.basename(json_file))[0]
    out_dir_3d = os.path.join(save_root, json_tag, "out_all")
    vel_dir = os.path.join(save_root, json_tag, "veolicity")
    diff0_dir = os.path.join(save_root, json_tag, "pred_y_0")
    diffnew_dir = os.path.join(save_root, json_tag, "pred_y_new")
    os.makedirs(out_dir_3d, exist_ok=True)
    os.makedirs(vel_dir, exist_ok=True)
    os.makedirs(diff0_dir, exist_ok=True)
    os.makedirs(diffnew_dir, exist_ok=True)
    if rank == 0:
        print(
            f"[INFO] start inference json={json_file}, n_windows={len(dataset)}, world_size={world_size}, batch_size_per_gpu={batch_size}",
            flush=True,
        )
    pbar = tqdm(loader, desc=f"infer {json_tag} rank{rank}", disable=rank != 0)
    with torch.no_grad():
        for batch_id, (
            x_data,
            x_mask,
            frame_idx,
            future_ts_batch,
            window_start_batch,
        ) in enumerate(pbar):
            x = x_data.to(device, non_blocking=True).float()
            x_mask = x_mask.to(device, non_blocking=True).float()
            t_steps = frame_idx.to(device, non_blocking=True).float()
            y_mask = make_y_mask_from_x_mask(x_mask, dataset.T_out)
            out = model(
                x,
                mode="nearest",
                x_mask=x_mask,
                y_mask=y_mask,
                lat_range=lat_range,
                lon_range=lon_range,
                t_steps=t_steps,
            )
            if isinstance(out, (tuple, list)):
                out_all = out[0]
                veolicity = out[1] if len(out) > 1 else None
            else:
                out_all = out
                veolicity = None
            x_zmax = torch.max(x, dim=2)[0]
            x_pre_zmax = torch.max(out_all, dim=2)[0]
            x_zmax_norm = diffusion_new.normalize(x_zmax)
            x_pre_zmax_norm = diffusion_new.normalize(x_pre_zmax)
            if y_mask.dim() == 6:
                y_mask_zmax = torch.max(y_mask, dim=2)[0]
            elif y_mask.dim() == 5:
                y_mask_zmax = torch.max(y_mask, dim=2)[0]
            else:
                raise ValueError(f"Unsupported y_mask dim: {y_mask.dim()}")
            pred_y_0, pred_y_new = run_two_diffusions(
                diffusion_0=diffusion_0,
                diffusion_new=diffusion_new,
                x_zmax_norm=x_zmax_norm,
                x_pre_zmax_norm=x_pre_zmax_norm,
                y_mask_zmax=y_mask_zmax,
                diff_threshold=diff_threshold,
                diff_min_size=diff_min_size,
            )
            B = x.shape[0]
            for b in range(B):
                future_ts = future_ts_batch[b]
                window_start = window_start_batch[b]
                if len(future_ts) != out_all.shape[1]:
                    raise ValueError(
                        f"future_ts length {len(future_ts)} != out_all T_out {out_all.shape[1]} for window {window_start}"
                    )
                for t, ts in enumerate(future_ts):
                    out3d_np = out_all[b, t].detach().cpu().numpy()
                    out3d_np = maybe_denorm(
                        out3d_np, save_denorm_out_all, normalize_div
                    )
                    save_one_array(
                        os.path.join(out_dir_3d, f"{ts}_out_all.npy"), out3d_np
                    )
                    if veolicity is not None:
                        vel_np = veolicity[b, t].detach().cpu().numpy()
                        save_one_array(
                            os.path.join(vel_dir, f"{ts}_veolicity.npy"), vel_np
                        )
                    pred0_np = pred_y_0[b, t].detach().cpu().numpy()
                    prednew_np = pred_y_new[b, t].detach().cpu().numpy()
                    pred0_np = maybe_denorm(pred0_np, save_denorm_pred2d, normalize_div)
                    prednew_np = maybe_denorm(
                        prednew_np, save_denorm_pred2d, normalize_div
                    )
                    save_one_array(
                        os.path.join(diff0_dir, f"{ts}_pred_y_0.npy"), pred0_np
                    )
                    save_one_array(
                        os.path.join(diffnew_dir, f"{ts}_pred_y_new.npy"), prednew_np
                    )
            if rank == 0 and batch_id == 0:
                print(
                    f"[INFO] first batch out_all shape={tuple(out_all.shape)}",
                    flush=True,
                )
                if veolicity is not None:
                    print(
                        f"[INFO] first batch veolicity shape={tuple(veolicity.shape)}",
                        flush=True,
                    )
                print(
                    f"[INFO] first batch pred_y_0 shape={tuple(pred_y_0.shape)}",
                    flush=True,
                )
                print(
                    f"[INFO] first batch pred_y_new shape={tuple(pred_y_new.shape)}",
                    flush=True,
                )
    if rank == 0:
        print(
            f"[INFO] done json={json_file}, saved under {os.path.join(save_root, json_tag)}",
            flush=True,
        )
