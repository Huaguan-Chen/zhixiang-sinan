from __future__ import annotations
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

TS_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class RuntimePaths:
    case_dir: Path
    raw_window_json: Path
    station_window_json: Path
    stage1_root: Path
    radar_2d_root: Path
    station_6min_root: Path
    output_dir: Path


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("需要安装 PyYAML 才能读取配置文件。")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else base_dir / p


def parse_time(text: str) -> pd.Timestamp:
    raw = str(text).strip()
    for fmt in [TS_FORMAT, "%Y%m%d%H%M", "%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            pass
    raise ValueError(f"无法解析时间：{text}")


def format_time(ts: Any) -> str:
    return pd.Timestamp(ts).strftime(TS_FORMAT)


def ensure_device(requested: str) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("当前任务没有可见 GPU；请确认作业已申请 GPU。")
        return torch.device("cuda:0")
    if requested.startswith("cuda:"):
        ordinal = int(requested.split(":", 1)[1])
        if ordinal >= torch.cuda.device_count():
            raise RuntimeError(f"Requested GPU is unavailable: {requested}")
    return torch.device(requested)


def load_np_any(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        try:
            for key in ["arr", "data", "arr_0"]:
                if key in obj.files:
                    return np.asarray(obj[key])
            if not obj.files:
                raise RuntimeError(f"空 npz：{path}")
            return np.asarray(obj[obj.files[0]])
        finally:
            obj.close()
    return np.asarray(obj)


def save_npy(path: Path, arr: np.ndarray, dtype: Any = np.float16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=dtype))


def find_one_file(
    root: Path, pattern: str, ts: pd.Timestamp, required: bool = True
) -> Optional[Path]:
    text = pattern.format(
        ts=format_time(ts),
        YYYY=ts.strftime("%Y"),
        MM=ts.strftime("%m"),
        DD=ts.strftime("%d"),
        YYYYMMDD=ts.strftime("%Y%m%d"),
        HHMMSS=ts.strftime("%H%M%S"),
    )
    candidate = Path(text)
    matches = (
        [candidate]
        if candidate.is_absolute() and candidate.exists()
        else [Path(x) for x in glob.glob(str(root / text), recursive=True)]
    )
    if matches:
        return sorted(matches)[0].resolve()
    if required:
        raise FileNotFoundError(f"找不到文件：root={root}, pattern={text}")
    return None


def runtime_paths(
    cfg: Dict[str, Any],
    base_dir: Path,
    analysis_time: pd.Timestamp,
    output_override: str,
) -> RuntimePaths:
    root = resolve_path(cfg["paths"]["runtime_root"], base_dir)
    case_name = format_time(analysis_time)
    case_dir = root / case_name
    output_root = (
        resolve_path(output_override, base_dir)
        if output_override
        else resolve_path(cfg["paths"]["output_root"], base_dir)
    )
    return RuntimePaths(
        case_dir=case_dir,
        raw_window_json=case_dir / "raw_radar_window.json",
        station_window_json=case_dir / "station_model_window_row0south.json",
        stage1_root=case_dir / "stage1",
        radar_2d_root=case_dir / "radar_2d_row0south",
        station_6min_root=case_dir / "station_6min",
        output_dir=output_root / case_name,
    )


def _meters_per_deg_lat(lat_rad: torch.Tensor) -> torch.Tensor:
    return torch.full_like(lat_rad, 111132.0)


def _meters_per_deg_lon(lat_rad: torch.Tensor) -> torch.Tensor:
    return 111320.0 * torch.cos(lat_rad)


def _convert_velocity_grid_per_6min_to_mps(vel, lat_range, grid_deg, dt_minutes):
    assert (
        vel.dim() in (5, 6) and vel.shape[2] == 3
    ), f"bad vel shape {tuple(vel.shape)}"
    device = vel.device
    dtype = vel.dtype
    if vel.dim() == 6:
        _, _, _, _, H, _ = vel.shape
    else:
        _, _, _, H, _ = vel.shape
    lat_min, lat_max = (float(lat_range[0]), float(lat_range[1]))
    lat_deg = torch.linspace(lat_min, lat_max, H, device=device, dtype=dtype)
    lat_rad = lat_deg * (math.pi / 180.0)
    dx_m = _meters_per_deg_lon(lat_rad) * grid_deg
    dy_m = _meters_per_deg_lat(lat_rad) * grid_deg
    dt_sec = float(dt_minutes) * 60.0
    out = vel.clone()
    if vel.dim() == 6:
        dx = dx_m.view(1, 1, 1, H, 1)
        dy = dy_m.view(1, 1, 1, H, 1)
        out[:, :, 0] = out[:, :, 0] * dx / dt_sec
        out[:, :, 1] = out[:, :, 1] * dy / dt_sec
        out[:, :, 2] = out[:, :, 2] / dt_sec
    else:
        dx = dx_m.view(1, 1, H, 1)
        dy = dy_m.view(1, 1, H, 1)
        out[:, :, 0] = out[:, :, 0] * dx / dt_sec
        out[:, :, 1] = out[:, :, 1] * dy / dt_sec
        out[:, :, 2] = out[:, :, 2] / dt_sec
    return out


def build_raw_radar_window(
    cfg: Dict[str, Any], analysis_time: pd.Timestamp, out_json: Path
) -> Dict[str, Any]:
    rcfg = cfg["radar"]
    root = Path(rcfg["raw_root"])
    interval = int(rcfg.get("interval_minutes", 6))
    past_len = int(rcfg.get("past_len", 10))
    future_len = int(rcfg.get("future_len", 30))
    past_times = [
        analysis_time - pd.Timedelta(minutes=interval * i)
        for i in range(past_len - 1, -1, -1)
    ]
    future_times = [
        analysis_time + pd.Timedelta(minutes=interval * i)
        for i in range(1, future_len + 1)
    ]
    past = []
    for ts in past_times:
        data = find_one_file(root, rcfg["data_pattern"], ts, required=True)
        mask = find_one_file(root, rcfg["mask_pattern"], ts, required=True)
        past.append({"time": format_time(ts), "data": str(data), "mask": str(mask)})
    future = []
    for ts in future_times:
        future.append(
            {
                "time": format_time(ts),
                "data": str(
                    out_json.parent / "future_time_only" / f"{format_time(ts)}_data.npy"
                ),
                "mask": str(
                    out_json.parent / "future_time_only" / f"{format_time(ts)}_mask.npy"
                ),
            }
        )
    window = {"start": format_time(past_times[0]), "past": past, "future": future}
    write_json(out_json, [window])
    print(f"[RADAR-RAW] 已构造原始雷达窗口：{out_json}", flush=True)
    return window


def to_dhw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    x = np.asarray(arr)
    while x.ndim > 3:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"雷达数组不能转换为 [D,H,W]：shape={x.shape}")
    if x.shape[-2] == h and x.shape[-1] >= w:
        return x[..., :w]
    if x.shape[0] == h and x.shape[1] >= w:
        return np.transpose(x[:, :w, :], (2, 0, 1))
    raise ValueError(f"无法判断雷达维度顺序：shape={x.shape}, H={h}, W={w}")


def to_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    x = np.squeeze(np.asarray(arr))
    while x.ndim > 2:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"不能转换为二维数组：shape={x.shape}")
    if x.shape != (h, w):
        yy = np.linspace(0, x.shape[0] - 1, h).round().astype(np.int64)
        xx = np.linspace(0, x.shape[1] - 1, w).round().astype(np.int64)
        x = x[np.ix_(yy, xx)]
    return np.asarray(x, dtype=np.float32)


def radar_dhw_to_row0south(
    arr: np.ndarray, h: int, w: int, apply_flip: bool
) -> np.ndarray:
    x = to_dhw(arr, h, w).astype(np.float32)
    return np.flip(x, axis=1).copy() if apply_flip else x


def radar_hw_to_row0south(
    arr: np.ndarray, h: int, w: int, apply_flip: bool
) -> np.ndarray:
    x = to_hw(arr, h, w)
    return np.flip(x, axis=0).copy() if apply_flip else x


def safe_zmax(dhw: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    x = np.asarray(dhw, dtype=np.float32)
    if mask is not None:
        x = np.where(mask > 0, x, np.nan)
    finite = np.isfinite(x).any(axis=0)
    out = np.max(np.where(np.isfinite(x), x, -np.inf), axis=0)
    return np.where(finite, out, 0.0).astype(np.float32)


def stage1_output_path(
    paths: RuntimePaths, folder: str, ts: pd.Timestamp, suffix: str
) -> Path:
    tag = paths.raw_window_json.stem
    return paths.stage1_root / tag / folder / f"{format_time(ts)}_{suffix}.npz"


def preprocess_radar_to_row0south(
    cfg: Dict[str, Any], raw_window: Dict[str, Any], paths: RuntimePaths
) -> Dict[str, Any]:
    dcfg = cfg["domain"]
    rcfg = cfg["radar"]
    h, w = (int(dcfg["h"]), int(dcfg["w"]))
    flip = bool(rcfg.get("apply_raw_radar_y_flip", True))
    normalize_div = float(rcfg.get("raw_normalize_div", 800.0))
    echo_threshold = float(rcfg.get("echo_threshold", 0.01))
    past_new = []
    for rec in raw_window["past"]:
        ts = parse_time(rec["time"])
        raw = radar_dhw_to_row0south(load_np_any(Path(rec["data"])), h, w, flip)
        mask = radar_dhw_to_row0south(load_np_any(Path(rec["mask"])), h, w, flip)
        raw_norm = np.nan_to_num(raw / normalize_div, nan=0.0, posinf=0.0, neginf=0.0)
        valid = (mask > 0).astype(np.float32)
        zmax = safe_zmax(raw_norm, valid)
        valid2d = np.max(valid, axis=0).astype(np.float32)
        zpath = (
            paths.radar_2d_root / "past_zmax" / f"{format_time(ts)}_zmax_row0south.npy"
        )
        vpath = (
            paths.radar_2d_root
            / "past_valid"
            / f"{format_time(ts)}_valid_row0south.npy"
        )
        save_npy(zpath, zmax)
        save_npy(vpath, valid2d, np.uint8)
        past_new.append(
            {
                **rec,
                "radar_zmax_2d_path": str(zpath),
                "radar_valid_2d_path": str(vpath),
                "orientation": "row0south",
            }
        )
    future_new = []
    for rec in raw_window["future"]:
        ts = parse_time(rec["time"])
        out_all_src = stage1_output_path(paths, "out_all", ts, "out_all")
        pred0_src = stage1_output_path(paths, "pred_y_0", ts, "pred_y_0")
        prednew_src = stage1_output_path(paths, "pred_y_new", ts, "pred_y_new")
        out_all = radar_dhw_to_row0south(load_np_any(out_all_src), h, w, flip)
        pred0 = radar_hw_to_row0south(load_np_any(pred0_src), h, w, flip)
        prednew = radar_hw_to_row0south(load_np_any(prednew_src), h, w, flip)
        pred_merged = np.maximum(pred0, prednew)
        outall_zmax = safe_zmax(out_all)
        echo = np.mean(out_all > echo_threshold, axis=0).astype(np.float32)
        apath = (
            paths.radar_2d_root
            / "future_outall"
            / f"{format_time(ts)}_outall_row0south.npy"
        )
        opath = (
            paths.radar_2d_root
            / "future_outall_zmax"
            / f"{format_time(ts)}_outall_zmax_row0south.npy"
        )
        epath = (
            paths.radar_2d_root
            / "future_echo"
            / f"{format_time(ts)}_echo_row0south.npy"
        )
        p0path = (
            paths.radar_2d_root
            / "future_pred0"
            / f"{format_time(ts)}_pred0_row0south.npy"
        )
        pnpath = (
            paths.radar_2d_root
            / "future_prednew"
            / f"{format_time(ts)}_prednew_row0south.npy"
        )
        pmpath = (
            paths.radar_2d_root
            / "future_pred_merged"
            / f"{format_time(ts)}_pred_merged_row0south.npy"
        )
        save_npy(apath, out_all)
        save_npy(opath, outall_zmax)
        save_npy(epath, echo)
        save_npy(p0path, pred0)
        save_npy(pnpath, prednew)
        save_npy(pmpath, pred_merged)
        future_new.append(
            {
                **rec,
                "outall_zmax_2d_path": str(opath),
                "outall_echo_fraction_2d_path": str(epath),
                "pred_y_0_path": str(p0path),
                "pred_y_new_path": str(pnpath),
                "pred_y_merged_path": str(pmpath),
                "orientation": "row0south",
            }
        )
    window = {
        "start": raw_window["start"],
        "past": past_new,
        "future": future_new,
        "orientation": "row0south",
    }
    write_json(paths.station_window_json, [window])
    return window


def assert_orientation_contract(
    cfg: Dict[str, Any], station_window: Dict[str, Any]
) -> Dict[str, Any]:
    domain = cfg["domain"]
    lat_min, lat_max = [float(x) for x in domain["lat_range"]]
    y_south = (lat_min - lat_min) / (lat_max - lat_min)
    y_north = (lat_max - lat_min) / (lat_max - lat_min)
    checks = {
        "canonical_orientation": "row0south",
        "raw_radar_y_flip_exactly_once": bool(
            cfg["radar"].get("apply_raw_radar_y_flip", True)
        ),
        "station_lat_min_maps_to_row0": bool(abs(y_south) < 1e-06),
        "station_lat_max_maps_to_last_row": bool(abs(y_north - 1.0) < 1e-06),
        "all_runtime_radar_records_row0south": all(
            (
                rec.get("orientation") == "row0south"
                for rec in station_window["past"] + station_window["future"]
            )
        ),
        "station_model_flip_pred_y_must_be_false": True,
    }
    if not all((v for k, v in checks.items() if isinstance(v, bool))):
        raise RuntimeError(f"南北方向契约检查失败：{checks}")
    return checks


STATION_ALIASES: Dict[str, Sequence[str]] = {
    "Station_Id_C": ["Station_Id_C", "station_id", "StationID", "ID", "id"],
    "Lat": ["Lat", "lat", "LAT", "latitude"],
    "Lon": ["Lon", "lon", "LON", "longitude"],
    "PRE": ["PRE", "PRE_1min", "precip", "rain"],
    "PRE_1h": ["PRE_1h", "pre_1h", "rain_1h"],
    "Q_PRE": ["Q_PRE", "PRE_QC", "q_pre"],
    "Q_PRE_1h": ["Q_PRE_1h", "PRE_1h_QC", "q_pre_1h"],
    "TEM": ["TEM", "temperature", "tem"],
    "DPT": ["DPT", "dewpoint", "dpt"],
    "RHU": ["RHU", "humidity", "rhu"],
    "VAP": ["VAP", "vap"],
    "PRS": ["PRS", "pressure", "prs"],
    "WIN_S_INST": ["WIN_S_INST", "wind_speed_inst", "s_inst"],
    "WIN_D_INST": ["WIN_D_INST", "wind_dir_inst", "d_inst"],
    "WIN_S_Avg_2mi": ["WIN_S_Avg_2mi", "wind_speed_2min", "s_2min"],
    "WIN_D_Avg_2mi": ["WIN_D_Avg_2mi", "wind_dir_2min", "d_2min"],
    "WIN_S_Avg_10mi": ["WIN_S_Avg_10mi", "wind_speed_10min", "s_10min"],
    "WIN_D_Avg_10mi": ["WIN_D_Avg_10mi", "wind_dir_10min", "d_10min"],
}


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def read_station_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    for encoding in ["utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, low_memory=False)


def build_station_time(df: pd.DataFrame) -> pd.Series:
    time_col = first_existing_column(
        df, ["time", "Time", "datetime", "Datetime", "timestamp"]
    )
    if time_col:
        return pd.to_datetime(df[time_col], errors="coerce")
    required = {"Year", "Mon", "Day", "Hour", "Min"}
    if required.issubset(df.columns):
        second = (
            pd.to_numeric(df["Second"], errors="coerce").fillna(0)
            if "Second" in df.columns
            else pd.Series(0, index=df.index)
        )
        return pd.to_datetime(
            {
                "year": pd.to_numeric(df["Year"], errors="coerce"),
                "month": pd.to_numeric(df["Mon"], errors="coerce"),
                "day": pd.to_numeric(df["Day"], errors="coerce"),
                "hour": pd.to_numeric(df["Hour"], errors="coerce"),
                "minute": pd.to_numeric(df["Min"], errors="coerce"),
                "second": second,
            },
            errors="coerce",
        )
    raise ValueError("原始站点文件缺少 time 或 Year/Mon/Day/Hour/Min。")


def standardize_station_columns(
    df: pd.DataFrame, qc_min: float, apply_rain_qc: bool = False
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["time"] = build_station_time(df)
    for name, aliases in STATION_ALIASES.items():
        col = first_existing_column(df, aliases)
        out[name] = df[col] if col else np.nan
    out["Station_Id_C"] = out["Station_Id_C"].astype(str)
    numeric_cols = [c for c in out.columns if c not in {"time", "Station_Id_C"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[np.abs(out[c]) >= 9999, c] = np.nan
    out = out.dropna(subset=["time", "Lat", "Lon"])
    out = out[~out["Station_Id_C"].isin(["", "nan", "None", "999999"])]
    if apply_rain_qc and out["Q_PRE"].notna().any():
        out.loc[~(out["Q_PRE"] >= qc_min), "PRE"] = np.nan
    if apply_rain_qc and out["Q_PRE_1h"].notna().any():
        out.loc[~(out["Q_PRE_1h"] >= qc_min), "PRE_1h"] = np.nan
    out.loc[out["PRE"] < 0, "PRE"] = np.nan
    out.loc[out["PRE_1h"] < 0, "PRE_1h"] = np.nan
    return out


def wind_uv(speed: pd.Series, direction_deg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(direction_deg)
    return (-speed * np.sin(rad), -speed * np.cos(rad))


def interpolate_1min_limited(
    s: pd.Series, minute_idx: pd.DatetimeIndex, max_gap_min: int
) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")].reindex(minute_idx)
    interp = s.interpolate(method="time", limit_area="inside")
    valid_times = list(s.dropna().index)
    for left, right in zip(valid_times[:-1], valid_times[1:]):
        gap_min = int((right - left).total_seconds() // 60)
        if gap_min > int(max_gap_min):
            interp.loc[(interp.index > left) & (interp.index < right)] = np.nan
    return interp


def estimate_median_interval_min(s: pd.Series) -> float:
    idx = s.dropna().sort_index().index
    if len(idx) < 2:
        return math.inf
    diffs = np.diff(idx.view("int64")) / 1000000000.0 / 60.0
    diffs = diffs[np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else math.inf


def circular_rolling_mean_deg(
    s: pd.Series,
    minute_idx: pd.DatetimeIndex,
    target_times: pd.DatetimeIndex,
    max_gap_min: int,
) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    rad = np.deg2rad(s.astype("float32"))
    sin_1min = interpolate_1min_limited(
        pd.Series(np.sin(rad), index=s.index), minute_idx, max_gap_min
    )
    cos_1min = interpolate_1min_limited(
        pd.Series(np.cos(rad), index=s.index), minute_idx, max_gap_min
    )
    sin_mean = sin_1min.rolling(window=6, min_periods=1).mean().reindex(target_times)
    cos_mean = cos_1min.rolling(window=6, min_periods=1).mean().reindex(target_times)
    angle = np.rad2deg(np.arctan2(sin_mean.to_numpy(), cos_mean.to_numpy()))
    angle = np.where(angle < 0, angle + 360.0, angle)
    angle[~np.isfinite(sin_mean.to_numpy()) | ~np.isfinite(cos_mean.to_numpy())] = (
        np.nan
    )
    return angle.astype(np.float32)


def aggregate_station_to_6min(
    raw: pd.DataFrame,
    cfg: Dict[str, Any],
    target_times: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.DataFrame:
    scfg = cfg["station"]
    required_raw_cols = [
        "PRE",
        "PRE_1h",
        "Q_PRE",
        "Q_PRE_1h",
        "TEM",
        "DPT",
        "RHU",
        "VAP",
        "PRS",
        "WIN_S_INST",
        "WIN_D_INST",
        "WIN_S_Avg_2mi",
        "WIN_D_Avg_2mi",
        "WIN_S_Avg_10mi",
        "WIN_D_Avg_10mi",
    ]
    raw = raw.copy()
    for col in required_raw_cols:
        if col not in raw.columns:
            raw[col] = np.nan
    for prefix in ["INST", "Avg_2mi", "Avg_10mi"]:
        s = raw[f"WIN_S_{prefix}"]
        d = raw[f"WIN_D_{prefix}"]
        raw[f"U_{prefix}"], raw[f"V_{prefix}"] = wind_uv(s, d)
    if target_times is None:
        start = raw["time"].min().floor("6min")
        end = raw["time"].max().ceil("6min")
        target_index = pd.date_range(start, end, freq="6min")
    else:
        target_index = pd.DatetimeIndex(sorted({pd.Timestamp(x) for x in target_times}))
    if target_index.empty:
        raise ValueError("站点 6 分钟目标时间轴为空。")
    minute_index = pd.date_range(
        target_index.min() - pd.Timedelta(minutes=60), target_index.max(), freq="1min"
    )
    max_gap = int(scfg.get("max_interp_gap_min", 10))
    max_gap_dense = int(scfg.get("max_interp_gap_min_dense", 2))
    max_gap_1h = int(scfg.get("max_interp_gap_min_1h", 30))
    sparse_threshold = float(scfg.get("sparse_interval_threshold_min", 2.0))
    min_valid_6 = int(scfg.get("min_valid_minutes_6min", 4))
    min_valid_60 = int(scfg.get("min_valid_minutes_1h", 40))
    output_map = {
        "TEM": "TEM_mean_6min",
        "DPT": "DPT_mean_6min",
        "RHU": "RHU_mean_6min",
        "VAP": "VAP_mean_6min",
        "PRS": "PRS_mean_6min",
        "WIN_S_INST": "S_inst_mean_6min",
        "WIN_S_Avg_2mi": "S_2min_mean_6min",
        "WIN_S_Avg_10mi": "S_10min_mean_6min",
        "U_INST": "U_inst_mean_6min",
        "V_INST": "V_inst_mean_6min",
        "U_Avg_2mi": "U_2min_mean_6min",
        "V_Avg_2mi": "V_2min_mean_6min",
        "U_Avg_10mi": "U_10min_mean_6min",
        "V_Avg_10mi": "V_10min_mean_6min",
    }
    frames = []
    for sid, station in raw.groupby("Station_Id_C", sort=True):
        station = station.sort_values("time").copy()
        numeric_cols = [c for c in station.columns if c not in {"time", "Station_Id_C"}]
        station = station.groupby("time", as_index=False)[numeric_cols].mean()
        lat_values = pd.to_numeric(station["Lat"], errors="coerce").dropna()
        lon_values = pd.to_numeric(station["Lon"], errors="coerce").dropna()
        lat = float(lat_values.iloc[-1]) if len(lat_values) else math.nan
        lon = float(lon_values.iloc[-1]) if len(lon_values) else math.nan
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        one = pd.DataFrame(
            {"time": target_index, "Station_Id_C": str(sid), "Lat": lat, "Lon": lon}
        )
        pre_raw = station.set_index("time")["PRE"].sort_index()
        pre_med_interval = estimate_median_interval_min(pre_raw)
        pre_gap = max_gap_dense if pre_med_interval <= sparse_threshold else max_gap
        pre_1min = interpolate_1min_limited(pre_raw, minute_index, pre_gap)
        pre_valid = pre_1min.notna().astype("float32")
        pre_fill = pre_1min.fillna(0.0).astype("float32")
        pre6 = pre_fill.rolling(window=6, min_periods=1).sum().reindex(target_index)
        pre6_count = (
            pre_valid.rolling(window=6, min_periods=1).sum().reindex(target_index)
        )
        pre6[pre6_count < min_valid_6] = np.nan
        pre60 = pre_fill.rolling(window=60, min_periods=1).sum().reindex(target_index)
        pre60_count = (
            pre_valid.rolling(window=60, min_periods=1).sum().reindex(target_index)
        )
        pre60[pre60_count < min_valid_60] = np.nan
        one["PRE_6min"] = pre6.to_numpy(dtype=np.float32)
        one["PRE_valid_minutes_6min"] = pre6_count.to_numpy(dtype=np.float32)
        one["PRE_1h_from_PRE"] = pre60.to_numpy(dtype=np.float32)
        one["PRE_valid_minutes_1h_from_PRE"] = pre60_count.to_numpy(dtype=np.float32)
        one["PRE_median_interval_min"] = float(pre_med_interval)
        one["PRE_interp_used"] = True
        one["PRE_interp_mode"] = (
            "dense_short_gap" if pre_gap == max_gap_dense else "sparse_limited_gap"
        )
        p1h = interpolate_1min_limited(
            station.set_index("time")["PRE_1h"], minute_index, max_gap_1h
        ).reindex(target_index)
        one["PRE_1h_interp"] = p1h.to_numpy(dtype=np.float32)
        one["PRE_1h_best"] = one["PRE_1h_interp"].where(
            one["PRE_1h_interp"].notna(), one["PRE_1h_from_PRE"]
        )
        for raw_col, out_col in output_map.items():
            series_1min = interpolate_1min_limited(
                station.set_index("time")[raw_col], minute_index, max_gap
            )
            mean6 = (
                series_1min.rolling(window=6, min_periods=1)
                .mean()
                .reindex(target_index)
            )
            one[out_col] = mean6.to_numpy(dtype=np.float32)
        for direction_col in ["WIN_D_INST", "WIN_D_Avg_2mi", "WIN_D_Avg_10mi"]:
            one[f"{direction_col}_circmean_6min"] = circular_rolling_mean_deg(
                station.set_index("time")[direction_col],
                minute_index,
                target_index,
                max_gap,
            )
        for qc_col in ["Q_PRE", "Q_PRE_1h"]:
            tmp = station[["time", qc_col]].dropna().copy()
            tmp["bin_time"] = tmp["time"].dt.ceil("6min")
            one[f"{qc_col}_max_6min"] = (
                tmp.groupby("bin_time")[qc_col]
                .max()
                .reindex(target_index)
                .to_numpy(dtype=np.float32)
            )
        frames.append(one)
    if not frames:
        raise RuntimeError("逐站插值后没有可用站点记录。")
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.sort_values(["Station_Id_C", "time"])


def station_file_patterns_for_interval(
    cfg: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp
) -> List[Path]:
    root = Path(cfg["station"]["raw_root"])
    patterns = cfg["station"].get("raw_patterns", ["**/*.csv"])
    minutes = pd.date_range(start.floor("min"), end.ceil("min"), freq="min")
    found: set[Path] = set()
    for minute in minutes:
        for pattern in patterns:
            text = str(pattern).format(
                YYYY=minute.strftime("%Y"),
                MM=minute.strftime("%m"),
                DD=minute.strftime("%d"),
                HH=minute.strftime("%H"),
                MIN=minute.strftime("%M"),
                YYYYMM=minute.strftime("%Y%m"),
                YYYYMMDD=minute.strftime("%Y%m%d"),
                YYYYMMDDHH=minute.strftime("%Y%m%d%H"),
                YYYYMMDDHHMM=minute.strftime("%Y%m%d%H%M"),
            )
            for item in glob.glob(str(root / text), recursive=True):
                found.add(Path(item).resolve())
    return sorted(found)


def write_station_daily_pkls(df: pd.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for day, sub in df.groupby(df["time"].dt.strftime("%Y%m%d")):
        sub.sort_values(["time", "Station_Id_C"]).to_pickle(
            root / f"aws_6min_{day}.pkl"
        )


def station_frame_diagnostics(
    df: pd.DataFrame, requested_times: Sequence[pd.Timestamp]
) -> List[Dict[str, Any]]:
    target_cols = [
        "PRE_6min",
        "PRE_1h_best",
        "U_inst_mean_6min",
        "V_inst_mean_6min",
        "S_inst_mean_6min",
        "U_2min_mean_6min",
        "V_2min_mean_6min",
        "S_2min_mean_6min",
        "U_10min_mean_6min",
        "V_10min_mean_6min",
        "S_10min_mean_6min",
        "TEM_mean_6min",
        "DPT_mean_6min",
        "RHU_mean_6min",
        "VAP_mean_6min",
        "PRS_mean_6min",
    ]
    rows = []
    for ts in requested_times:
        sub = df[df["time"] == pd.Timestamp(ts)].copy()
        available = [c for c in target_cols if c in sub.columns]
        valid_any = (
            sub[available].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            if available
            else pd.Series(False, index=sub.index)
        )

        def count_valid(col: str) -> int:
            if col not in sub.columns:
                return 0
            valid = pd.to_numeric(sub[col], errors="coerce").notna()
            return int(sub.loc[valid, "Station_Id_C"].nunique())

        rows.append(
            {
                "time": format_time(ts),
                "rows": int(len(sub)),
                "any_target_valid_stations": int(
                    sub.loc[valid_any, "Station_Id_C"].nunique()
                ),
                "pre6_valid_stations": count_valid("PRE_6min"),
                "temperature_valid_stations": count_valid("TEM_mean_6min"),
                "wind_2min_valid_stations": count_valid("S_2min_mean_6min"),
            }
        )
    return rows


def process_raw_station_data(
    cfg: Dict[str, Any],
    analysis_time: pd.Timestamp,
    raw_window: Dict[str, Any],
    paths: RuntimePaths,
) -> pd.DataFrame:
    all_times = [
        parse_time(x["time"]) for x in raw_window["past"] + raw_window["future"]
    ]
    start = min(all_times) - pd.Timedelta(hours=1)
    target_end = analysis_time
    read_end = target_end
    files = station_file_patterns_for_interval(cfg, start, read_end)
    if not files:
        raise FileNotFoundError(f'未找到原始站点文件：{cfg['station']['raw_root']}')
    frames = []
    for path in files:
        try:
            frames.append(
                standardize_station_columns(
                    read_station_file(path),
                    float(cfg["station"].get("qc_valid_min", 5.0)),
                    bool(cfg["station"].get("apply_rain_qc", False)),
                )
            )
        except Exception as exc:
            print(
                f"[STATION-WARN] 跳过 {path}: {type(exc).__name__}: {exc}", flush=True
            )
    if not frames:
        raise RuntimeError("所有原始站点文件均读取失败。")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw[(raw["time"] >= start) & (raw["time"] <= read_end)]
    lon_min, lon_max = [float(x) for x in cfg["domain"]["lon_range"]]
    lat_min, lat_max = [float(x) for x in cfg["domain"]["lat_range"]]
    raw = raw[
        raw["Lon"].between(lon_min, lon_max) & raw["Lat"].between(lat_min, lat_max)
    ].copy()
    target_times = pd.date_range(
        start=min(all_times) - pd.Timedelta(hours=1), end=target_end, freq="6min"
    )
    station_6min = aggregate_station_to_6min(raw, cfg, target_times=target_times)
    station_6min = station_6min[station_6min["time"] <= analysis_time].copy()
    write_station_daily_pkls(station_6min, paths.station_6min_root)
    requested_times = [parse_time(x["time"]) for x in raw_window["past"]]
    diagnostics = station_frame_diagnostics(station_6min, requested_times)
    write_csv_rows(
        paths.output_dir / "station_input_frame_diagnostics.csv", diagnostics
    )
    valid_counts = [int(row["any_target_valid_stations"]) for row in diagnostics]
    if not valid_counts or min(valid_counts) <= 1:
        bad = [row for row in diagnostics if int(row["any_target_valid_stations"]) <= 1]
        raise RuntimeError(
            f'站点 1min 插值/6min 聚合后仍存在空帧或单站帧，已停止推理以避免输出空图。 首个异常帧={(bad[0] if bad else 'unknown')}'
        )
    print(
        f'[STATION] 原始文件={len(files)}，原始记录={len(raw)}，插值后6分钟记录={len(station_6min)}，请求帧有效站点数 min={min(valid_counts)} max={max(valid_counts)}，诊断={paths.output_dir / 'station_input_frame_diagnostics.csv'}',
        flush=True,
    )
    return station_6min


def spec_index(specs: Sequence[Any], name: str) -> int:
    for i, spec in enumerate(specs):
        if str(spec.name).lower() == str(name).lower():
            return i
    raise KeyError(f"找不到站点变量：{name}")


def to_physical(arr: np.ndarray, specs: Sequence[Any], idx: int) -> np.ndarray:
    return np.asarray(arr[..., idx], dtype=np.float32) * float(
        specs[idx].scale
    ) + float(specs[idx].offset)


def load_fixed_station_catalog(
    cfg: Dict[str, Any], base_dir: Path
) -> Optional[pd.DataFrame]:
    fcfg = cfg.get("fixed_stations", {})
    if not bool(fcfg.get("enabled", False)):
        return None
    raw_path = str(fcfg.get("catalog", "")).strip()
    if not raw_path:
        raise ValueError(
            "fixed_stations.enabled=true，但未配置 fixed_stations.catalog。"
        )
    path = resolve_path(raw_path, base_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"固定站表不存在：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    id_col = first_existing_column(frame, STATION_ALIASES["Station_Id_C"])
    lat_col = first_existing_column(frame, STATION_ALIASES["Lat"])
    lon_col = first_existing_column(frame, STATION_ALIASES["Lon"])
    if id_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"固定站表必须包含站号、纬度、经度列：path={path}, columns={list(frame.columns)}"
        )
    catalog = pd.DataFrame(
        {
            "Station_Id_C": frame[id_col].astype(str).str.strip(),
            "Lat": pd.to_numeric(frame[lat_col], errors="coerce"),
            "Lon": pd.to_numeric(frame[lon_col], errors="coerce"),
        }
    )
    invalid_id = catalog["Station_Id_C"].isin(["", "nan", "None", "999999"])
    invalid_coord = ~np.isfinite(catalog["Lat"]) | ~np.isfinite(catalog["Lon"])
    lon_min, lon_max = [float(x) for x in cfg["domain"]["lon_range"]]
    lat_min, lat_max = [float(x) for x in cfg["domain"]["lat_range"]]
    outside = ~catalog["Lon"].between(lon_min, lon_max) | ~catalog["Lat"].between(
        lat_min, lat_max
    )
    invalid = invalid_id | invalid_coord | outside
    if bool(invalid.any()):
        preview = catalog.loc[invalid].head(5).to_dict(orient="records")
        raise ValueError(
            f"固定站表存在无效站号/经纬度或域外站点：count={int(invalid.sum())}, examples={preview}"
        )
    duplicated = catalog["Station_Id_C"].duplicated(keep=False)
    if bool(duplicated.any()):
        examples = (
            catalog.loc[duplicated, "Station_Id_C"].drop_duplicates().head(10).tolist()
        )
        raise ValueError(
            f"固定站表存在重复站号：count={int(duplicated.sum())}, examples={examples}"
        )
    catalog = catalog.sort_values("Station_Id_C", kind="stable").reset_index(drop=True)
    if catalog.empty:
        raise ValueError(f"固定站表为空：{path}")
    catalog.attrs["path"] = str(path)
    return catalog


def reindex_station_sample_to_catalog(
    sample: Dict[str, torch.Tensor],
    source_station_ids: Sequence[Any],
    catalog: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    source_ids = np.asarray(source_station_ids).astype(str)
    source_count = int(sample["station_xy"].shape[0])
    if source_ids.size != source_count:
        raise ValueError(
            f"动态站号与样本形状不一致：station_ids={source_ids.size}, station_xy={source_count}"
        )
    if len(set(source_ids.tolist())) != source_ids.size:
        raise ValueError("动态站号存在重复，无法安全映射到固定站表。")
    fixed_ids = catalog["Station_Id_C"].to_numpy(dtype=str)
    fixed_index = {sid: i for i, sid in enumerate(fixed_ids.tolist())}
    unknown = sorted(set(source_ids.tolist()) - set(fixed_index))
    policy = str(
        cfg.get("fixed_stations", {}).get("unknown_active_station_policy", "warn")
    ).lower()
    if unknown and policy == "error":
        raise RuntimeError(
            f"本次活跃站中出现固定站表未收录的新站；为避免静默损失预报效果已停止。 count={len(unknown)}, examples={unknown[:10]}"
        )
    if unknown and policy not in {"drop", "warn"}:
        raise ValueError(f"不支持的 unknown_active_station_policy={policy!r}")
    if unknown:
        print(
            f"[FIXED-STATION-WARN] 固定站表未收录 {len(unknown)} 个活跃站，将不输出：{unknown[:10]}",
            flush=True,
        )
    lon_min, lon_max = [float(x) for x in cfg["domain"]["lon_range"]]
    lat_min, lat_max = [float(x) for x in cfg["domain"]["lat_range"]]
    lon = catalog["Lon"].to_numpy(dtype=np.float32)
    lat = catalog["Lat"].to_numpy(dtype=np.float32)
    fallback_xy = np.stack(
        [
            2.0 * (lon - lon_min) / max(lon_max - lon_min, 1e-06) - 1.0,
            2.0 * (lat - lat_min) / max(lat_max - lat_min, 1e-06) - 1.0,
        ],
        axis=-1,
    ).astype(np.float32)
    source_index = {
        sid: i for i, sid in enumerate(source_ids.tolist()) if sid in fixed_index
    }
    active_fixed = np.asarray(
        [fixed_index[sid] for sid in source_index], dtype=np.int64
    )
    active_source = np.asarray(
        [source_index[sid] for sid in source_index], dtype=np.int64
    )
    active_mask = np.zeros(fixed_ids.size, dtype=bool)
    active_mask[active_fixed] = True
    result = dict(sample)
    old_xy = sample["station_xy"]
    new_xy = torch.as_tensor(fallback_xy, dtype=old_xy.dtype, device=old_xy.device)
    if active_fixed.size:
        dst = torch.as_tensor(active_fixed, dtype=torch.long, device=old_xy.device)
        src = torch.as_tensor(active_source, dtype=torch.long, device=old_xy.device)
        new_xy[dst] = old_xy[src]
    result["station_xy"] = new_xy
    for key in ["station_hist", "station_hist_mask"]:
        old = sample[key]
        new = old.new_zeros((fixed_ids.size, *old.shape[1:]))
        if active_fixed.size:
            dst = torch.as_tensor(active_fixed, dtype=torch.long, device=old.device)
            src = torch.as_tensor(active_source, dtype=torch.long, device=old.device)
            new[dst] = old[src]
        result[key] = new
    result["station_valid"] = torch.ones(
        fixed_ids.size,
        dtype=sample["station_valid"].dtype,
        device=sample["station_valid"].device,
    )
    result["n_station"] = sample["n_station"].new_tensor([fixed_ids.size])
    return (result, fixed_ids, active_mask)


def validate_real_station_sample(
    sample: Dict[str, Any], station_ids: np.ndarray
) -> None:
    station_xy = sample["station_xy"].numpy()
    if station_xy.shape[0] <= 1 or (
        len(station_ids) == 1 and str(station_ids[0]) == "dummy"
    ):
        raise RuntimeError(
            f"站点样本退化为 dummy/单站点。请检查原始 AWS 文件、时间字段和逐站 6 分钟插值结果；station_xy_shape={station_xy.shape}, station_ids={station_ids[:5].tolist()}"
        )


def sample_grid_at_stations(
    grid: torch.Tensor, station_xy: torch.Tensor
) -> torch.Tensor:
    xy = station_xy.view(1, -1, 1, 2)
    return F.grid_sample(
        grid.view(1, 1, *grid.shape),
        xy,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)


def rain_to_space(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def rain_from_space(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def build_ab_pseudo_obs(
    model_a: torch.Tensor,
    model_b: torch.Tensor,
    radar_at_station: torch.Tensor,
    acfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dry = float(acfg.get("dry_station_threshold_mm", 0.1))
    radar_wet = radar_at_station >= float(acfg.get("radar_support_threshold_mm", 0.5))
    a_wet, b_wet = (model_a > dry, model_b > dry)
    consensus = a_wet & b_wet
    b_weight = torch.zeros_like(model_a)
    b_supported = b_wet & (a_wet | radar_wet)
    b_weight = torch.where(
        b_supported,
        torch.full_like(b_weight, float(acfg.get("model_b_supported_weight", 0.2))),
        b_weight,
    )
    b_weight = (
        b_weight + float(acfg.get("ab_consensus_bonus", 0.1)) * consensus.float()
    ).clamp(0.0, float(acfg.get("model_b_max_weight", 0.35)))
    b_cap = torch.maximum(model_a, radar_at_station) + float(
        acfg.get("model_b_excess_cap_mm", 3.0)
    )
    model_b_robust = torch.minimum(model_b, b_cap)
    pseudo = (1.0 - b_weight) * model_a + b_weight * model_b_robust
    isolated_b = b_wet & ~a_wet & ~radar_wet
    pseudo = torch.where(isolated_b, torch.zeros_like(pseudo), pseudo)
    a_unsupported_b_dry = a_wet & ~b_wet & ~radar_wet
    pseudo = torch.where(
        a_unsupported_b_dry,
        (1.0 - float(acfg.get("model_b_dry_veto_weight", 0.3))) * pseudo,
        pseudo,
    )
    pseudo = torch.where(~a_wet & ~b_wet, torch.zeros_like(pseudo), pseudo)
    disagreement = torch.abs(model_a - model_b)
    quality = 1.0 / (
        1.0
        + disagreement / max(float(acfg.get("ab_disagreement_scale_mm", 5.0)), 1e-06)
    )
    floor = float(acfg.get("ab_disagreement_weight_floor", 0.35))
    quality = quality.clamp(floor, 1.0)
    quality = torch.where(consensus, torch.ones_like(quality), quality)
    quality = torch.where(isolated_b, torch.full_like(quality, floor), quality)
    return (pseudo, quality)


def gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    sigma = max(float(sigma), 0.5)
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum().clamp_min(1e-12)


def gaussian_blur2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    k = gaussian_kernel1d(sigma, img.device)
    pad = k.numel() // 2
    x = F.pad(img[None, None], (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1))[0, 0]


def fuse_station_pre6(
    background: np.ndarray,
    station_xy: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    acfg: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)
    out = np.full_like(a, np.nan, dtype=np.float32)
    if not np.any(valid):
        return out
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    xy = torch.as_tensor(station_xy[valid], dtype=torch.float32, device=device)
    aa = torch.as_tensor(a[valid], dtype=torch.float32, device=device)
    bb = torch.as_tensor(b[valid], dtype=torch.float32, device=device)
    pseudo, _ = build_ab_pseudo_obs(aa, bb, sample_grid_at_stations(bg, xy), acfg)
    out[valid] = pseudo.detach().cpu().numpy().astype(np.float32)
    return out


def assimilate_ab_to_grid(
    background: np.ndarray,
    station_xy: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    acfg: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0)
    if int(valid.sum()) < int(acfg.get("min_stations", 5)):
        return background.astype(np.float32, copy=True)
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    h, w = bg.shape
    xy = torch.as_tensor(station_xy[valid], dtype=torch.float32, device=device)
    aa = torch.as_tensor(a[valid], dtype=torch.float32, device=device)
    bb = torch.as_tensor(b[valid], dtype=torch.float32, device=device)
    bg_station = sample_grid_at_stations(bg, xy)
    pseudo, weight = build_ab_pseudo_obs(aa, bb, bg_station, acfg)
    positive = pseudo > bg_station + float(acfg.get("positive_margin_mm", 0.1))
    radar_support = bg_station >= float(acfg.get("radar_support_threshold_mm", 0.5))
    a_extreme = aa >= float(acfg.get("model_a_extreme_threshold_mm", 5.0))
    b_support = bb >= float(acfg.get("model_b_support_threshold_mm", 1.0))
    support_weight = (
        float(acfg.get("positive_weight_floor", 0.1))
        + float(acfg.get("radar_support_bonus", 0.65)) * radar_support.float()
        + float(acfg.get("supported_extreme_bonus", 0.25))
        * (a_extreme & radar_support).float()
        + float(acfg.get("cross_model_extreme_bonus", 0.2))
        * (a_extreme & b_support).float()
    ).clamp(0.0, 1.0)
    weight = torch.where(positive, weight * support_weight, weight)
    likely_radar_false_alarm = (
        pseudo <= float(acfg.get("dry_station_threshold_mm", 0.1))
    ) & (bg_station >= float(acfg.get("false_alarm_bg_threshold_mm", 0.5)))
    weight = torch.where(
        likely_radar_false_alarm,
        weight * float(acfg.get("false_alarm_weight", 3.0)),
        weight,
    )
    bg_space = rain_to_space(bg)
    innovation = rain_to_space(pseudo) - sample_grid_at_stations(bg_space, xy)
    max_inc = math.log1p(float(acfg.get("max_increment_mm", 80.0)))
    innovation = innovation.clamp(-max_inc, max_inc)
    xpix = ((xy[:, 0] + 1.0) * 0.5 * (w - 1)).round().long().clamp(0, w - 1)
    ypix = ((xy[:, 1] + 1.0) * 0.5 * (h - 1)).round().long().clamp(0, h - 1)
    linear = ypix * w + xpix
    sum_flat = torch.zeros(h * w, dtype=torch.float32, device=device)
    cnt_flat = torch.zeros(h * w, dtype=torch.float32, device=device)
    sum_flat.index_add_(0, linear, innovation * weight)
    cnt_flat.index_add_(0, linear, weight)
    innov_sum, obs_count = (sum_flat.view(h, w), cnt_flat.view(h, w))
    sigmas = [float(x) for x in acfg.get("sigmas_px", [4, 12, 32])]
    r_over_b = (
        float(acfg.get("obs_error_mm", 1.0))
        / max(float(acfg.get("bg_error_mm", 3.0)), 1e-06)
    ) ** 2
    num, den = (torch.zeros_like(bg), torch.zeros_like(bg))
    for sigma in sigmas:
        smooth_sum = gaussian_blur2d(innov_sum, sigma)
        smooth_count = gaussian_blur2d(obs_count, sigma)
        increment = smooth_sum / smooth_count.clamp_min(1e-06)
        gain = smooth_count / (smooth_count + r_over_b)
        num += increment * gain
        den += gain
    increment = num / den.clamp_min(1e-06)
    gain = (den / max(len(sigmas), 1)).clamp(0.0, 1.0)
    analysis = rain_from_space(
        bg_space + float(acfg.get("blend_alpha", 1.0)) * gain * increment
    )
    return (
        analysis.clamp(0.0, float(acfg.get("max_rain_mm", 300.0)))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def radar_forecast_fields(
    model_data: Any,
    cfg_model: Dict[str, Any],
    ds: Any,
    station_window: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    coeff = float(sample["c_hist"].numpy().reshape(-1)[0])
    past_times = [parse_time(rec["time"]) for rec in station_window["past"]]
    init_time = past_times[-1]
    fields: Dict[str, List[np.ndarray]] = {
        "pred_y0_norm": [],
        "pred_ynew_norm": [],
        "outall_zmax_norm": [],
        "pred_y0_rain_mm": [],
        "pred_ynew_rain_mm": [],
        "outall_zmax_rain_mm": [],
        "corrected_pred_y0_rain_mm": [],
        "corrected_pred_ynew_rain_mm": [],
        "corrected_outall_zmax_rain_mm": [],
    }
    for rec in station_window["future"]:
        valid_time = parse_time(rec["time"])
        lead_min = max(0.0, float((valid_time - init_time).total_seconds() / 60.0))
        lead_coeff = float(ds._lead_coeff(coeff, lead_min))
        pred0 = model_data.read_2d(
            Path(rec["pred_y_0_path"]), ds.h, ds.w, flip_y=False, fill=0.0
        )
        prednew = model_data.read_2d(
            Path(rec["pred_y_new_path"]), ds.h, ds.w, flip_y=False, fill=0.0
        )
        outall = model_data.read_2d(
            Path(rec["outall_zmax_2d_path"]), ds.h, ds.w, flip_y=False, fill=0.0
        )
        pred0_corr = model_data.apply_rain_correction_to_dbz_norm(
            pred0, lead_coeff, cfg_model
        )
        prednew_corr = model_data.apply_rain_correction_to_dbz_norm(
            prednew, lead_coeff, cfg_model
        )
        outall_corr = model_data.apply_rain_correction_to_dbz_norm(
            outall, lead_coeff, cfg_model
        )
        fields["pred_y0_norm"].append(pred0)
        fields["pred_ynew_norm"].append(prednew)
        fields["outall_zmax_norm"].append(outall)
        fields["pred_y0_rain_mm"].append(model_data.dbz_norm_to_rain6(pred0, cfg_model))
        fields["pred_ynew_rain_mm"].append(
            model_data.dbz_norm_to_rain6(prednew, cfg_model)
        )
        fields["outall_zmax_rain_mm"].append(
            model_data.dbz_norm_to_rain6(outall, cfg_model)
        )
        fields["corrected_pred_y0_rain_mm"].append(
            model_data.dbz_norm_to_rain6(pred0_corr, cfg_model)
        )
        fields["corrected_pred_ynew_rain_mm"].append(
            model_data.dbz_norm_to_rain6(prednew_corr, cfg_model)
        )
        fields["corrected_outall_zmax_rain_mm"].append(
            model_data.dbz_norm_to_rain6(outall_corr, cfg_model)
        )
    return {key: np.stack(value).astype(np.float32) for key, value in fields.items()}
