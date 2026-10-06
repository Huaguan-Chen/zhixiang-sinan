from __future__ import annotations
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_TIME_PATTERNS = ["(20\\d{6}_\\d{6})", "(20\\d{10})", "(20\\d{8})"]


def parse_time_from_string(s: str) -> Optional[pd.Timestamp]:
    if not isinstance(s, str):
        return None
    for pat in _TIME_PATTERNS:
        m = re.search(pat, s)
        if not m:
            continue
        t = m.group(1)
        try:
            if "_" in t:
                return pd.Timestamp(datetime.strptime(t, "%Y%m%d_%H%M%S"))
            if len(t) == 12:
                return pd.Timestamp(datetime.strptime(t, "%Y%m%d%H%M"))
            if len(t) == 10:
                return pd.Timestamp(datetime.strptime(t, "%Y%m%d%H"))
            if len(t) == 8:
                return pd.Timestamp(datetime.strptime(t, "%Y%m%d"))
        except Exception:
            pass
    return None


def parse_time_from_record(rec: Dict[str, Any]) -> Optional[pd.Timestamp]:
    for key in [
        "time",
        "timestamp",
        "ts",
        "datetime",
        "radar_zmax_2d_path",
        "outall_zmax_2d_path",
        "pred_y_0_path",
        "pred0_path",
        "pred_y_new_path",
        "prednew_path",
        "grid_path",
        "data",
        "mask",
        "out_all_path",
        "veolicity_path",
    ]:
        v = rec.get(key)
        if isinstance(v, str):
            t = parse_time_from_string(v)
            if t is not None:
                return t
    for v in rec.values():
        if isinstance(v, str):
            t = parse_time_from_string(v)
            if t is not None:
                return t
    return None


def resolve_path(p: Optional[str], root_dir: str = "") -> Optional[Path]:
    if not p:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    if not root_dir:
        return pp
    root = Path(root_dir)
    candidates = [root / pp, root.parent / pp, pp]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def resolve_first_existing_path(
    rec: Dict[str, Any], keys: Sequence[str], root_dir: str = ""
) -> Optional[Path]:
    first = None
    for k in keys:
        v = rec.get(k)
        if not v:
            continue
        q = resolve_path(v, root_dir)
        if first is None:
            first = q
        if q is not None and q.exists():
            return q
    return first


def load_np_any(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        try:
            if "arr" in obj.files:
                arr = obj["arr"]
            elif "data" in obj.files:
                arr = obj["data"]
            else:
                if len(obj.files) == 0:
                    raise KeyError(f"empty npz file: {path}")
                arr = obj[obj.files[0]]
        finally:
            obj.close()
    else:
        arr = obj
    return np.asarray(arr)


def reduce_to_2d_array(arr: np.ndarray, h: int, w: int, path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[-2:] == (h, w):
            arr = arr[0]
        elif arr.shape[-1] <= 8 and arr.shape[:2] == (h, w):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    else:
        while arr.ndim > 2:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"cannot reduce to 2D: path={path}, shape={arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape != (h, w):
        yy = np.linspace(0, arr.shape[0] - 1, h).round().astype(np.int64)
        xx = np.linspace(0, arr.shape[1] - 1, w).round().astype(np.int64)
        arr = arr[np.ix_(yy, xx)]
    return arr.astype(np.float32, copy=False)


def read_2d(
    path: Optional[Path], h: int, w: int, flip_y: bool = False, fill: float = 0.0
) -> np.ndarray:
    if path is None or not path.exists():
        raise FileNotFoundError(f"2D input file not found: {path}")
    arr = reduce_to_2d_array(load_np_any(path), h, w, path)
    if flip_y:
        arr = arr[::-1, :]
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr.astype(np.float32, copy=False)


def make_time_features(ts: pd.Timestamp) -> np.ndarray:
    ts = pd.Timestamp(ts)
    minute_of_day = ts.hour * 60 + ts.minute
    day_of_year = ts.dayofyear
    return np.asarray(
        [
            math.sin(2.0 * math.pi * minute_of_day / 1440.0),
            math.cos(2.0 * math.pi * minute_of_day / 1440.0),
            math.sin(2.0 * math.pi * day_of_year / 366.0),
            math.cos(2.0 * math.pi * day_of_year / 366.0),
            ts.hour / 23.0,
            ts.minute / 59.0 if ts.minute > 0 else 0.0,
        ],
        dtype=np.float32,
    )


def add_constant_maps(
    channels: List[np.ndarray], values: Sequence[float], h: int, w: int
) -> None:
    for v in values:
        channels.append(np.full((h, w), float(v), dtype=np.float32))


def lonlat_to_xy_norm(
    lon: np.ndarray, lat: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_min = float(cfg["lon_min"])
    lon_max = float(cfg["lon_max"])
    lat_min = float(cfg["lat_min"])
    lat_max = float(cfg["lat_max"])
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-06)
    if bool(cfg.get("north_to_south_y", True)):
        y = (lat_max - lat) / max(lat_max - lat_min, 1e-06)
    else:
        y = (lat - lat_min) / max(lat_max - lat_min, 1e-06)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= 1.0)
        & (y >= 0.0)
        & (y <= 1.0)
    )
    gx = x * 2.0 - 1.0
    gy = y * 2.0 - 1.0
    return (
        x.astype(np.float32),
        y.astype(np.float32),
        gx.astype(np.float32),
        gy.astype(np.float32),
        valid,
    )


def sample_grid_np(
    arr: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray
) -> np.ndarray:
    h, w = (arr.shape[-2], arr.shape[-1])
    ix = np.clip(np.round(x_norm * (w - 1)).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(y_norm * (h - 1)).astype(np.int64), 0, h - 1)
    return arr[iy, ix].astype(np.float32)


def z_to_rain_rate_mm_h(
    dbz: np.ndarray, a: float = 200.0, b: float = 1.6
) -> np.ndarray:
    z_lin = np.power(10.0, dbz / 10.0)
    r = np.power(np.maximum(z_lin / max(a, 1e-06), 0.0), 1.0 / max(b, 1e-06))
    return r.astype(np.float32)


def dbz_norm_to_rain6(norm: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    zr = cfg.get("zr", {})
    dbz_scale = float(zr.get("dbz_scale", 80.0))
    dbz_offset = float(zr.get("dbz_offset", 0.0))
    a = float(zr.get("a", 200.0))
    b = float(zr.get("b", 1.6))
    minutes = float(zr.get("accum_minutes", 6.0))
    dbz = np.asarray(norm, dtype=np.float32) * dbz_scale + dbz_offset
    dbz = np.clip(dbz, float(zr.get("dbz_min", 0.0)), float(zr.get("dbz_max", 80.0)))
    rain_rate = z_to_rain_rate_mm_h(dbz, a=a, b=b)
    rain6 = rain_rate * (minutes / 60.0)
    rain6 = np.nan_to_num(rain6, nan=0.0, posinf=0.0, neginf=0.0)
    return rain6.astype(np.float32)


def apply_rain_correction_to_dbz_norm(
    norm: np.ndarray, coeff: float, cfg: Dict[str, Any]
) -> np.ndarray:
    zr = cfg.get("zr", {})
    b = float(zr.get("b", 1.6))
    dbz_scale = float(zr.get("dbz_scale", 80.0))
    c = max(float(coeff), 1e-06)
    dbz_shift = 10.0 * b * math.log10(c)
    out = np.asarray(norm, dtype=np.float32) + dbz_shift / max(dbz_scale, 1e-06)
    return np.clip(out, 0.0, 1.5).astype(np.float32)


def station_id_column(df: pd.DataFrame) -> str:
    for c in ["Station_Id_C", "station_id", "StationID", "Station_Id_d", "id", "ID"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find station id column")


@dataclass
class VarSpec:
    name: str
    candidates: List[str]
    offset: float
    scale: float
    nonnegative: bool = False


VARIABLE_SPECS = [
    VarSpec("pre6", ["PRE_6min"], 0.0, 25.0, True),
    VarSpec(
        "pre1h",
        [
            "PRE_1h_best",
            "PRE_1h_interp",
            "PRE_1h_from_PRE",
            "PRE_1h_mean_6min",
            "PRE_1h",
        ],
        0.0,
        100.0,
        True,
    ),
    VarSpec(
        "u_inst",
        ["U_inst_mean_6min", "WIN_S_INST_U_mean_6min", "U_inst", "u_inst"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "v_inst",
        ["V_inst_mean_6min", "WIN_S_INST_V_mean_6min", "V_inst", "v_inst"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "s_inst",
        ["S_inst_mean_6min", "WIN_S_INST_mean_6min", "S_inst", "s_inst"],
        0.0,
        40.0,
        True,
    ),
    VarSpec(
        "u_2min",
        ["U_2min_mean_6min", "WIN_S_Avg_2mi_U_mean_6min", "U_2min", "u_2min"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "v_2min",
        ["V_2min_mean_6min", "WIN_S_Avg_2mi_V_mean_6min", "V_2min", "v_2min"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "s_2min",
        ["S_2min_mean_6min", "WIN_S_Avg_2mi_mean_6min", "S_2min", "s_2min"],
        0.0,
        40.0,
        True,
    ),
    VarSpec(
        "u_10min",
        ["U_10min_mean_6min", "WIN_S_Avg_10mi_U_mean_6min", "U_10min", "u_10min"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "v_10min",
        ["V_10min_mean_6min", "WIN_S_Avg_10mi_V_mean_6min", "V_10min", "v_10min"],
        0.0,
        40.0,
        False,
    ),
    VarSpec(
        "s_10min",
        ["S_10min_mean_6min", "WIN_S_Avg_10mi_mean_6min", "S_10min", "s_10min"],
        0.0,
        40.0,
        True,
    ),
    VarSpec("tem", ["TEM_mean_6min", "TEM", "tem"], 0.0, 50.0, False),
    VarSpec("dpt", ["DPT_mean_6min", "DPT", "dpt"], 0.0, 50.0, False),
    VarSpec("rhu", ["RHU_mean_6min", "RHU", "rhu"], 0.0, 100.0, True),
    VarSpec("vap", ["VAP_mean_6min", "VAP", "vap"], 0.0, 60.0, True),
    VarSpec("prs", ["PRS_mean_6min", "PRS", "prs"], 1000.0, 100.0, False),
]
HISTORY_VAR_NAMES = [
    "pre6",
    "pre1h",
    "u_inst",
    "v_inst",
    "s_inst",
    "u_2min",
    "v_2min",
    "s_2min",
    "u_10min",
    "v_10min",
    "s_10min",
    "tem",
    "dpt",
    "rhu",
    "vap",
    "prs",
]


class Station6Cache:
    def __init__(self, root: str, max_days: int = 16):
        self.root = Path(root)
        self.max_days = int(max_days)
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.group_cache: OrderedDict[str, Dict[np.int64, pd.DataFrame]] = OrderedDict()

    def _path_for_day(self, day: pd.Timestamp) -> Path:
        return self.root / f"aws_6min_{day:%Y%m%d}.pkl"

    def get_day(self, day: pd.Timestamp) -> pd.DataFrame:
        key = f"{day:%Y%m%d}"
        if key in self.cache:
            df = self.cache.pop(key)
            self.cache[key] = df
            return df
        path = self._path_for_day(day)
        if not path.exists():
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_pickle(path)
                if "time" in df.columns:
                    df = df.copy()
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                    df = df.dropna(subset=["time"])
                    if not df.empty:
                        sid = station_id_column(df)
                        df[sid] = df[sid].astype(str)
                else:
                    df = pd.DataFrame()
            except Exception as e:
                print(
                    f"[WARN] failed to read station day {path}: {type(e).__name__}: {e}",
                    flush=True,
                )
                df = pd.DataFrame()
        self.cache[key] = df
        while len(self.cache) > self.max_days:
            old_key, _ = self.cache.popitem(last=False)
            self.group_cache.pop(old_key, None)
        return df

    def get_day_groups(self, day: pd.Timestamp) -> Dict[np.int64, pd.DataFrame]:
        key = f"{day:%Y%m%d}"
        if key in self.group_cache:
            groups = self.group_cache.pop(key)
            self.group_cache[key] = groups
            return groups
        df = self.get_day(day)
        groups: Dict[np.int64, pd.DataFrame] = {}
        if not df.empty and "time" in df.columns:
            for t, sub in df.groupby("time", sort=False):
                groups[np.int64(pd.Timestamp(t).value)] = sub.copy()
        self.group_cache[key] = groups
        while len(self.group_cache) > self.max_days:
            self.group_cache.popitem(last=False)
        return groups

    def get_frame(self, ts: pd.Timestamp) -> pd.DataFrame:
        ts = (
            pd.Timestamp(ts).tz_localize(None)
            if getattr(pd.Timestamp(ts), "tzinfo", None)
            else pd.Timestamp(ts)
        )
        day = pd.Timestamp(datetime(ts.year, ts.month, ts.day))
        groups = self.get_day_groups(day)
        key = np.int64(ts.value)
        out = groups.get(key)
        if out is not None:
            return out.copy()
        if groups:
            keys = np.asarray(list(groups.keys()), dtype=np.int64)
            diffs = np.abs(keys - key)
            j = int(np.argmin(diffs))
            if diffs[j] <= np.int64(pd.Timedelta(seconds=1).value):
                return groups[np.int64(keys[j])].copy()
        return pd.DataFrame()


class StationForecastDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        paths = cfg["paths"]
        json_path = paths["window_json"]
        self.root_dir = paths.get("root_dir", "")
        self.aws_root = paths["aws_6min_root"]
        self.station_cache = Station6Cache(
            self.aws_root,
            max_days=int(cfg.get("data", {}).get("station_day_cache", 16)),
        )
        self.items = self._load_windows(json_path)
        dcfg = cfg["data"]
        self.h = int(dcfg.get("h", 512))
        self.w = int(dcfg.get("w", 512))
        self.past_len = int(dcfg.get("past_len", 10))
        self.future_len = int(dcfg.get("future_len", 30))
        self.flip_pred_y = bool(dcfg.get("flip_pred_y", True))
        self.variable_specs = VARIABLE_SPECS
        self.variable_names = [v.name for v in self.variable_specs]
        self.var_by_name = {v.name: v for v in self.variable_specs}
        self.hist_names = [v for v in HISTORY_VAR_NAMES if v in self.var_by_name]
        self.wind_diag_names = [
            "conv_inst",
            "vort_inst",
            "mfc_inst",
            "conv_2min",
            "vort_2min",
            "mfc_2min",
            "conv_10min",
            "vort_10min",
            "mfc_10min",
        ]
        self.domain_cfg = cfg["domain"]
        self.rain_scale = self.var_by_name["pre6"].scale
        self.time_feature_dim = 6
        self.coord_channels = self._make_coord_channels()
        cache_size = int(dcfg.get("frame_cache_size", 256))
        self.frame_array_cache: OrderedDict[np.int64, Dict[str, Any]] = OrderedDict()
        self.frame_grid_cache: OrderedDict[np.int64, Tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self.frame_cache_size = max(16, cache_size)
        pfcfg = cfg.get("persistent_frame_cache", {})
        self.persistent_frame_cache_enabled = bool(pfcfg.get("enabled", False))
        self.persistent_frame_cache_root = (
            Path(str(pfcfg.get("root", ""))) if pfcfg.get("root", "") else None
        )
        self.persistent_frame_cache_strict = bool(pfcfg.get("strict", False))

    def _load_windows(self, json_path: str) -> List[Dict[str, Any]]:
        p = Path(json_path)
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for key in ["windows", "samples", "data", "items"]:
                if key in obj and isinstance(obj[key], list):
                    obj = obj[key]
                    break
        if not isinstance(obj, list):
            raise ValueError(f"Unsupported json structure: {json_path}")
        items = []
        for it in obj:
            if isinstance(it, dict) and "past" in it and ("future" in it):
                items.append(it)
        if not items:
            raise ValueError(f"No windows found in {json_path}")
        return items

    def _make_coord_channels(self) -> Tuple[np.ndarray, np.ndarray]:
        yy = np.linspace(-1.0, 1.0, self.h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, self.w, dtype=np.float32)[None, :]
        return (
            np.broadcast_to(xx, (self.h, self.w)).copy(),
            np.broadcast_to(yy, (self.h, self.w)).copy(),
        )

    def _get_times(
        self, recs: List[Dict[str, Any]], expected_len: int
    ) -> List[pd.Timestamp]:
        times = [parse_time_from_record(r) for r in recs]
        if all((t is not None for t in times)):
            return [pd.Timestamp(t) for t in times[:expected_len]]
        good = [t for t in times if t is not None]
        if good:
            first_idx = next((i for i, t in enumerate(times) if t is not None))
            base = pd.Timestamp(times[first_idx]) - pd.Timedelta(minutes=6 * first_idx)
        else:
            base = pd.Timestamp("1970-01-01")
        filled = [base + pd.Timedelta(minutes=6 * i) for i in range(len(recs))]
        if len(filled) < expected_len:
            filled += [
                filled[-1] + pd.Timedelta(minutes=6 * (i + 1))
                for i in range(expected_len - len(filled))
            ]
        return filled[:expected_len]

    def _prepare_window_records_times(self, idx: int):
        w = self.items[idx]
        past_recs = list(w.get("past", []))[-self.past_len :]
        future_recs = list(w.get("future", []))[: self.future_len]
        if len(past_recs) < self.past_len:
            past_recs = [past_recs[0]] * (self.past_len - len(past_recs)) + past_recs
        if len(future_recs) < self.future_len:
            future_recs = future_recs + [future_recs[-1]] * (
                self.future_len - len(future_recs)
            )
        past_times = self._get_times(past_recs, self.past_len)
        future_times = self._get_times(future_recs, self.future_len)
        return (past_recs, future_recs, past_times, future_times)

    def window_start_string(self, idx: int) -> str:
        w = self.items[idx]
        for key in ["start", "start_time", "window_start", "time"]:
            if key in w:
                return str(w[key])
        try:
            _, _, past_times, _ = self._prepare_window_records_times(idx)
            return pd.Timestamp(past_times[-1]).strftime("%Y%m%d_%H%M%S")
        except Exception:
            return f"idx{idx:06d}"

    def _read_past_zmax(self, rec: Dict[str, Any]) -> np.ndarray:
        p = resolve_first_existing_path(rec, ["radar_zmax_2d_path"], self.root_dir)
        return read_2d(p, self.h, self.w, flip_y=False, fill=0.0)

    def _read_future_arrays(
        self, rec: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0 = resolve_first_existing_path(
            rec, ["pred_y_0_path", "pred0_path"], self.root_dir
        )
        pn = resolve_first_existing_path(
            rec, ["pred_y_new_path", "prednew_path"], self.root_dir
        )
        px = resolve_first_existing_path(rec, ["outall_zmax_2d_path"], self.root_dir)
        pe = resolve_first_existing_path(
            rec, ["outall_echo_fraction_2d_path"], self.root_dir
        )
        missing = []
        if p0 is None or not p0.exists():
            missing.append(("pred_y_0_path/pred0_path", p0))
        if pn is None or not pn.exists():
            missing.append(("pred_y_new_path/prednew_path", pn))
        if px is None or not px.exists():
            missing.append(("outall_zmax_2d_path", px))
        if pe is None or not pe.exists():
            missing.append(("outall_echo_fraction_2d_path", pe))
        if missing:
            msg = "; ".join([f"{k} -> {v}" for k, v in missing])
            raise FileNotFoundError(
                f"missing future radar input files: {msg}. record keys={list(rec.keys())}"
            )
        pred0 = read_2d(p0, self.h, self.w, flip_y=self.flip_pred_y, fill=0.0)
        prednew = read_2d(pn, self.h, self.w, flip_y=self.flip_pred_y, fill=0.0)
        xpre = read_2d(px, self.h, self.w, flip_y=False, fill=0.0)
        echo = read_2d(pe, self.h, self.w, flip_y=False, fill=0.0)
        return (pred0, prednew, xpre, echo)

    def _extract_values(self, df: pd.DataFrame, spec: VarSpec) -> np.ndarray:
        for c in spec.candidates:
            if c in df.columns:
                arr = pd.to_numeric(df[c], errors="coerce").to_numpy(
                    dtype=np.float32, copy=True
                )
                arr[~np.isfinite(arr)] = np.nan
                return arr
        return np.full(len(df), np.nan, dtype=np.float32)

    def _value_from_row(self, row: pd.Series, spec: VarSpec) -> float:
        for c in spec.candidates:
            if c in row.index:
                try:
                    v = float(
                        pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                    )
                    if np.isfinite(v):
                        return v
                except Exception:
                    pass
        return np.nan

    def _station_coords(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if df.empty or "Lon" not in df.columns or "Lat" not in df.columns:
            n = len(df)
            z = np.full(n, np.nan, dtype=np.float32)
            return (z, z, z, z, np.zeros(n, dtype=bool))
        lon = pd.to_numeric(df["Lon"], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        lat = pd.to_numeric(df["Lat"], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        return lonlat_to_xy_norm(lon, lat, self.domain_cfg)

    def _cache_put_lru(self, cache: OrderedDict, key: np.int64, value: Any) -> Any:
        cache[key] = value
        while len(cache) > self.frame_cache_size:
            cache.popitem(last=False)
        return value

    def _frame_cache_path(self, ts: pd.Timestamp) -> Optional[Path]:
        if (
            not self.persistent_frame_cache_enabled
            or self.persistent_frame_cache_root is None
        ):
            return None
        ts = pd.Timestamp(ts)
        return self.persistent_frame_cache_root / f"day_{ts:%Y%m%d}.pkl"

    def _load_persistent_frame_arrays(
        self, ts: pd.Timestamp
    ) -> Optional[Dict[str, Any]]:
        path = self._frame_cache_path(ts)
        if path is None:
            return None
        if not path.exists():
            if self.persistent_frame_cache_strict:
                raise FileNotFoundError(
                    f"persistent daily station frame cache missing: {path}"
                )
            return None
        if not hasattr(self, "_persistent_day_cache"):
            self._persistent_day_cache = OrderedDict()
        day_key = str(path)
        if day_key in self._persistent_day_cache:
            day_obj = self._persistent_day_cache.pop(day_key)
            self._persistent_day_cache[day_key] = day_obj
        else:
            day_obj = pd.read_pickle(path)
            self._persistent_day_cache[day_key] = day_obj
            while len(self._persistent_day_cache) > int(
                self.cfg.get("data", {}).get("station_day_cache", 16)
            ):
                self._persistent_day_cache.popitem(last=False)
        key = pd.Timestamp(ts).strftime("%Y%m%d_%H%M%S")
        item = day_obj.get(key)
        if item is None:
            if self.persistent_frame_cache_strict:
                raise KeyError(
                    f"timestamp {key} missing in daily station frame cache {path}"
                )
            return None
        return {
            "sids": np.asarray(item["sids"]).astype(str),
            "x_norm": np.asarray(item["x_norm"], dtype=np.float32),
            "y_norm": np.asarray(item["y_norm"], dtype=np.float32),
            "gx": np.asarray(item["gx"], dtype=np.float32),
            "gy": np.asarray(item["gy"], dtype=np.float32),
            "values": np.asarray(item["values"], dtype=np.float32),
            "masks": np.asarray(item["masks"], dtype=np.float32),
        }

    def build_frame_arrays_from_pkl(self, ts: pd.Timestamp) -> Dict[str, Any]:
        df = self.station_cache.get_frame(ts)
        C = len(self.variable_specs)
        empty = {
            "sids": np.asarray([], dtype=str),
            "x_norm": np.zeros(0, dtype=np.float32),
            "y_norm": np.zeros(0, dtype=np.float32),
            "gx": np.zeros(0, dtype=np.float32),
            "gy": np.zeros(0, dtype=np.float32),
            "values": np.zeros((0, C), dtype=np.float32),
            "masks": np.zeros((0, C), dtype=np.float32),
        }
        if df.empty:
            return empty
        sid_col = station_id_column(df)
        df = df.copy()
        df[sid_col] = df[sid_col].astype(str)
        x_norm, y_norm, gx, gy, valid_xy = self._station_coords(df)
        if not np.any(valid_xy):
            return empty
        n = len(df)
        values = np.zeros((n, C), dtype=np.float32)
        masks = np.zeros((n, C), dtype=np.float32)
        for ci, spec in enumerate(self.variable_specs):
            raw = self._extract_values(df, spec)
            ok = np.isfinite(raw)
            if spec.nonnegative:
                ok = ok & (raw >= 0.0)
            values[ok, ci] = ((raw[ok] - spec.offset) / max(spec.scale, 1e-06)).astype(
                np.float32
            )
            masks[ok, ci] = 1.0
        keep = valid_xy & (masks.sum(axis=1) > 0)
        if not np.any(keep):
            return empty
        return {
            "sids": df[sid_col].to_numpy(dtype=str)[keep],
            "x_norm": x_norm[keep].astype(np.float32),
            "y_norm": y_norm[keep].astype(np.float32),
            "gx": gx[keep].astype(np.float32),
            "gy": gy[keep].astype(np.float32),
            "values": values[keep].astype(np.float32),
            "masks": masks[keep].astype(np.float32),
        }

    def _frame_arrays(self, ts: pd.Timestamp) -> Dict[str, Any]:
        key = np.int64(pd.Timestamp(ts).value)
        if key in self.frame_array_cache:
            item = self.frame_array_cache.pop(key)
            self.frame_array_cache[key] = item
            return item
        item = self._load_persistent_frame_arrays(ts)
        if item is not None:
            return self._cache_put_lru(self.frame_array_cache, key, item)
        item = self.build_frame_arrays_from_pkl(ts)
        return self._cache_put_lru(self.frame_array_cache, key, item)

    def _variable_index(self, name: str) -> int:
        return self.variable_names.index(name)

    def _mean_grid_from_frame_arrays(
        self, fa: Dict[str, Any], name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        value = np.zeros((self.h, self.w), dtype=np.float32)
        count = np.zeros((self.h, self.w), dtype=np.float32)
        if fa["sids"].size == 0:
            return (value, count)
        try:
            ci = self._variable_index(name)
        except ValueError:
            return (value, count)
        vals = np.asarray(fa["values"][:, ci], dtype=np.float32)
        masks = np.asarray(fa["masks"][:, ci], dtype=np.float32) > 0
        if not np.any(masks):
            return (value, count)
        x_norm = np.asarray(fa["x_norm"], dtype=np.float32)
        y_norm = np.asarray(fa["y_norm"], dtype=np.float32)
        ix = np.clip(
            np.round(x_norm[masks] * (self.w - 1)).astype(np.int64), 0, self.w - 1
        )
        iy = np.clip(
            np.round(y_norm[masks] * (self.h - 1)).astype(np.int64), 0, self.h - 1
        )
        v = vals[masks]
        np.add.at(value, (iy, ix), v)
        np.add.at(count, (iy, ix), 1.0)
        value = np.divide(
            value, np.maximum(count, 1.0), out=np.zeros_like(value), where=count > 0
        ).astype(np.float32)
        return (value, count)

    def _rasterize_station_frame_arrays(self, fa: Dict[str, Any]) -> np.ndarray:
        nvar = len(self.hist_names)
        out = np.zeros((nvar + 3, self.h, self.w), dtype=np.float32)
        if fa["sids"].size == 0:
            return out
        x_norm = np.asarray(fa["x_norm"], dtype=np.float32)
        y_norm = np.asarray(fa["y_norm"], dtype=np.float32)
        ix_all = np.clip(
            np.round(x_norm * (self.w - 1)).astype(np.int64), 0, self.w - 1
        )
        iy_all = np.clip(
            np.round(y_norm * (self.h - 1)).astype(np.int64), 0, self.h - 1
        )
        station_count = np.zeros((self.h, self.w), dtype=np.float32)
        np.add.at(station_count, (iy_all, ix_all), 1.0)
        out[nvar + 2] = np.log1p(station_count)
        cell = iy_all * self.w + ix_all
        for vi, name in enumerate(self.hist_names):
            try:
                ci = self._variable_index(name)
            except ValueError:
                continue
            vals = np.asarray(fa["values"][:, ci], dtype=np.float32)
            masks = np.asarray(fa["masks"][:, ci], dtype=np.float32) > 0
            if not np.any(masks):
                continue
            norm_vals = vals[masks]
            cells = cell[masks]
            uniq, inv = np.unique(cells, return_inverse=True)
            counts = np.bincount(inv).astype(np.float32)
            sums = np.bincount(inv, weights=norm_vals).astype(np.float32)
            means = sums / np.maximum(counts, 1.0)
            yy = uniq // self.w
            xx = uniq % self.w
            out[vi, yy, xx] = means
            if name == "pre6":
                maxs = np.full(len(uniq), -np.inf, dtype=np.float32)
                np.maximum.at(maxs, inv, norm_vals)
                sq = np.bincount(inv, weights=norm_vals * norm_vals).astype(np.float32)
                stds = np.sqrt(
                    np.maximum(sq / np.maximum(counts, 1.0) - means * means, 0.0)
                )
                out[nvar + 0, yy, xx] = maxs
                out[nvar + 1, yy, xx] = stds
        return out.astype(np.float32)

    @staticmethod
    def _box_smooth_masked(
        value: np.ndarray, count: np.ndarray, iters: int = 4
    ) -> np.ndarray:
        mask = (count > 0).astype(np.float32)
        num = value.astype(np.float32) * mask
        den = mask.copy()
        for _ in range(max(1, int(iters))):
            num_p = np.pad(num, ((1, 1), (1, 1)), mode="edge")
            den_p = np.pad(den, ((1, 1), (1, 1)), mode="edge")
            num = (
                num_p[:-2, :-2]
                + num_p[:-2, 1:-1]
                + num_p[:-2, 2:]
                + num_p[1:-1, :-2]
                + num_p[1:-1, 1:-1]
                + num_p[1:-1, 2:]
                + num_p[2:, :-2]
                + num_p[2:, 1:-1]
                + num_p[2:, 2:]
            )
            den = (
                den_p[:-2, :-2]
                + den_p[:-2, 1:-1]
                + den_p[:-2, 2:]
                + den_p[1:-1, :-2]
                + den_p[1:-1, 1:-1]
                + den_p[1:-1, 2:]
                + den_p[2:, :-2]
                + den_p[2:, 1:-1]
                + den_p[2:, 2:]
            )
        return np.divide(
            num, np.maximum(den, 1e-06), out=np.zeros_like(num), where=den > 1e-06
        ).astype(np.float32)

    def _wind_diagnostic_maps_from_frame_arrays(self, fa: Dict[str, Any]) -> np.ndarray:
        out = np.zeros((len(self.wind_diag_names), self.h, self.w), dtype=np.float32)
        if fa["sids"].size == 0:
            return out
        if "vap" in self.variable_names:
            vap, vap_count = self._mean_grid_from_frame_arrays(fa, "vap")
            vap_s = self._box_smooth_masked(vap, vap_count, iters=4)
        else:
            vap_s = np.ones((self.h, self.w), dtype=np.float32)
        wind_pairs = [
            ("u_inst", "v_inst", 0),
            ("u_2min", "v_2min", 3),
            ("u_10min", "v_10min", 6),
        ]
        for u_name, v_name, base in wind_pairs:
            if u_name not in self.variable_names or v_name not in self.variable_names:
                continue
            u, uc = self._mean_grid_from_frame_arrays(fa, u_name)
            v, vc = self._mean_grid_from_frame_arrays(fa, v_name)
            c = np.maximum(uc, vc)
            if not np.any(c > 0):
                continue
            u_s = self._box_smooth_masked(u, c, iters=4)
            v_s = self._box_smooth_masked(v, c, iters=4)
            du_dy, du_dx = np.gradient(u_s)
            dv_dy, dv_dx = np.gradient(v_s)
            conv = -du_dx + dv_dy
            vort = dv_dx + du_dy
            uq = u_s * vap_s
            vq = v_s * vap_s
            duq_dy, duq_dx = np.gradient(uq)
            dvq_dy, dvq_dx = np.gradient(vq)
            mfc = -duq_dx + dvq_dy
            out[base + 0] = np.clip(conv, -5.0, 5.0).astype(np.float32)
            out[base + 1] = np.clip(vort, -5.0, 5.0).astype(np.float32)
            out[base + 2] = np.clip(mfc, -5.0, 5.0).astype(np.float32)
        return out.astype(np.float32)

    def _frame_grid_maps(self, ts: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        key = np.int64(pd.Timestamp(ts).value)
        if key in self.frame_grid_cache:
            item = self.frame_grid_cache.pop(key)
            self.frame_grid_cache[key] = item
            return item
        fa = self._frame_arrays(ts)
        station_maps = self._rasterize_station_frame_arrays(fa)
        wind_diag_maps = self._wind_diagnostic_maps_from_frame_arrays(fa)
        item = (station_maps, wind_diag_maps)
        return self._cache_put_lru(self.frame_grid_cache, key, item)

    def _compute_zr_correction(
        self, past_times: List[pd.Timestamp], past_zmax: List[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        zcfg = self.cfg.get("dynamic_zr_correction", {})
        if not bool(zcfg.get("enabled", True)):
            return (
                1.0,
                {"valid_pairs": 0.0, "station_sum": 0.0, "radar_sum": 0.0, "raw": 1.0},
            )
        station_sum = 0.0
        radar_sum = 0.0
        valid_pairs = 0
        pre_spec = self.var_by_name["pre6"]
        for ts, zmax in zip(past_times, past_zmax):
            df = self.station_cache.get_frame(ts)
            if df.empty:
                continue
            x_norm, y_norm, _, _, valid_xy = self._station_coords(df)
            pre = self._extract_values(df, pre_spec)
            ok = valid_xy & np.isfinite(pre) & (pre >= 0.0)
            if not np.any(ok):
                continue
            z_sample = sample_grid_np(zmax, x_norm[ok], y_norm[ok])
            rr = dbz_norm_to_rain6(z_sample, self.cfg)
            p = pre[ok]
            pair_ok = (
                np.isfinite(rr)
                & np.isfinite(p)
                & (
                    (rr >= float(zcfg.get("pair_min_radar_mm", 0.02)))
                    | (p >= float(zcfg.get("pair_min_station_mm", 0.02)))
                )
            )
            if not np.any(pair_ok):
                continue
            station_sum += float(np.nansum(p[pair_ok]))
            radar_sum += float(np.nansum(rr[pair_ok]))
            valid_pairs += int(np.sum(pair_ok))
        min_pairs = int(zcfg.get("min_valid_pairs", 30))
        min_station_sum = float(zcfg.get("min_station_sum_mm", 1.0))
        min_radar_sum = float(zcfg.get("min_radar_sum_mm", 1.0))
        if (
            valid_pairs < min_pairs
            or station_sum < min_station_sum
            or radar_sum < min_radar_sum
        ):
            c = 1.0
            raw = 1.0
        else:
            raw = station_sum / max(radar_sum, 1e-06)
            c = float(
                np.clip(
                    raw,
                    float(zcfg.get("clip_min", 0.65)),
                    float(zcfg.get("clip_max", 1.35)),
                )
            )
        return (
            c,
            {
                "valid_pairs": float(valid_pairs),
                "station_sum": station_sum,
                "radar_sum": radar_sum,
                "raw": float(raw),
            },
        )

    def _lead_coeff(self, c_hist: float, lead_min: float) -> float:
        zcfg = self.cfg.get("dynamic_zr_correction", {})
        if not bool(zcfg.get("lead_decay", False)):
            return float(c_hist)
        tau = float(zcfg.get("tau_minutes", 60.0))
        alpha = math.exp(-float(lead_min) / max(tau, 1e-06))
        return float(1.0 + alpha * (float(c_hist) - 1.0))

    def compute_window_correction(self, idx: int) -> Dict[str, Any]:
        past_recs, future_recs, past_times, future_times = (
            self._prepare_window_records_times(idx)
        )
        past_zmax = [self._read_past_zmax(r) for r in past_recs]
        c_hist, info = self._compute_zr_correction(past_times, past_zmax)
        init_time = past_times[-1]
        lead_coeffs = []
        lead_minutes = []
        for ts in future_times[: self.future_len]:
            lead_min = max(0.0, float((ts - init_time).total_seconds() / 60.0))
            lead_minutes.append(lead_min)
            lead_coeffs.append(self._lead_coeff(c_hist, lead_min))
        return {
            "idx": int(idx),
            "start": self.window_start_string(idx),
            "c_hist": float(c_hist),
            "c_raw": float(info.get("raw", c_hist)),
            "station_sum": float(info.get("station_sum", 0.0)),
            "radar_sum": float(info.get("radar_sum", 0.0)),
            "valid_pairs": float(info.get("valid_pairs", 0.0)),
            "lead_coeffs": np.asarray(lead_coeffs, dtype=np.float32),
            "lead_minutes": np.asarray(lead_minutes, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        past_recs, future_recs, past_times, future_times = (
            self._prepare_window_records_times(idx)
        )
        past_zmax = [self._read_past_zmax(r) for r in past_recs]
        c_hist, c_info = self._compute_zr_correction(past_times, past_zmax)
        grid_dtype = str(self.cfg.get("data", {}).get("grid_dtype", "float16")).lower()
        np_grid_dtype = (
            np.float16 if grid_dtype in ["fp16", "float16", "half"] else np.float32
        )
        past_ch = (
            2
            + len(self.hist_names)
            + 3
            + len(self.wind_diag_names)
            + self.time_feature_dim
            + 2
        )
        past_grid = np.empty(
            (self.past_len, past_ch, self.h, self.w), dtype=np_grid_dtype
        )
        for ti, (ts, zmax) in enumerate(zip(past_times, past_zmax)):
            zr = dbz_norm_to_rain6(zmax, self.cfg) / self.rain_scale
            station_maps, wind_diag_maps = self._frame_grid_maps(ts)
            ch = 0
            past_grid[ti, ch] = zmax.astype(np_grid_dtype, copy=False)
            ch += 1
            past_grid[ti, ch] = zr.astype(np_grid_dtype, copy=False)
            ch += 1
            n = station_maps.shape[0]
            past_grid[ti, ch : ch + n] = station_maps.astype(np_grid_dtype, copy=False)
            ch += n
            n = wind_diag_maps.shape[0]
            past_grid[ti, ch : ch + n] = wind_diag_maps.astype(
                np_grid_dtype, copy=False
            )
            ch += n
            tf = make_time_features(ts)
            for v in tf:
                past_grid[ti, ch].fill(float(v))
                ch += 1
            past_grid[ti, ch] = self.coord_channels[0].astype(np_grid_dtype, copy=False)
            ch += 1
            past_grid[ti, ch] = self.coord_channels[1].astype(np_grid_dtype, copy=False)
            ch += 1
            if ch != past_ch:
                raise RuntimeError(f"past_ch mismatch: filled {ch}, expected {past_ch}")
        future_ch = 25
        future_grid = np.empty(
            (self.future_len, future_ch, self.h, self.w), dtype=np_grid_dtype
        )
        init_time = past_times[-1]
        for j, (ts, rec) in enumerate(zip(future_times, future_recs)):
            pred0, prednew, xpre, echo = self._read_future_arrays(rec)
            predmax = np.maximum(pred0, prednew).astype(np.float32)
            lead_min = max(0.0, float((ts - init_time).total_seconds() / 60.0))
            c_lead = self._lead_coeff(c_hist, lead_min)
            pred0_corr = apply_rain_correction_to_dbz_norm(pred0, c_lead, self.cfg)
            prednew_corr = apply_rain_correction_to_dbz_norm(prednew, c_lead, self.cfg)
            predmax_corr = apply_rain_correction_to_dbz_norm(predmax, c_lead, self.cfg)
            xpre_corr = apply_rain_correction_to_dbz_norm(xpre, c_lead, self.cfg)
            zr0 = dbz_norm_to_rain6(pred0, self.cfg)
            zrnew = dbz_norm_to_rain6(prednew, self.cfg)
            zrmax = dbz_norm_to_rain6(predmax, self.cfg)
            zr0_corr = dbz_norm_to_rain6(pred0_corr, self.cfg)
            zrnew_corr = dbz_norm_to_rain6(prednew_corr, self.cfg)
            zrmax_corr = dbz_norm_to_rain6(predmax_corr, self.cfg)
            ch = 0
            future_grid[j, ch] = pred0.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = prednew.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = predmax.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = xpre.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = echo.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = pred0_corr.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = prednew_corr.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = predmax_corr.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = xpre_corr.astype(np_grid_dtype, copy=False)
            ch += 1
            future_grid[j, ch] = (zr0 / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = (zrnew / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = (zrmax / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = (zr0_corr / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = (zrnew_corr / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = (zrmax_corr / self.rain_scale).astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch].fill(float(c_lead))
            ch += 1
            future_grid[j, ch].fill(float(lead_min / 180.0))
            ch += 1
            tf = make_time_features(ts)
            for v in tf:
                future_grid[j, ch].fill(float(v))
                ch += 1
            future_grid[j, ch] = self.coord_channels[0].astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            future_grid[j, ch] = self.coord_channels[1].astype(
                np_grid_dtype, copy=False
            )
            ch += 1
            if ch != future_ch:
                raise RuntimeError(
                    f"future_ch mismatch: filled {ch}, expected {future_ch}"
                )
        if bool(self.cfg.get("data", {}).get("compact_future_rain_only", False)):
            future_grid = future_grid[
                :, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], :, :
            ]
        days = pd.date_range(
            (past_times[0] - pd.Timedelta(hours=1)).normalize(),
            past_times[-1].normalize(),
            freq="D",
        )
        roster = pd.concat(
            [self.station_cache.get_day(day) for day in days], ignore_index=True
        )
        roster = (
            roster[roster["time"] <= past_times[-1]]
            .sort_values("time")
            .groupby("Station_Id_C", as_index=False)
            .tail(1)
        )
        _, _, gx, gy, inside = self._station_coords(roster)
        roster = roster.loc[inside].copy()
        roster["gx"] = gx[inside]
        roster["gy"] = gy[inside]
        roster = roster.sort_values("Station_Id_C")
        station_order = roster["Station_Id_C"].to_numpy(dtype=str)
        if station_order.size <= 1:
            raise RuntimeError("Insufficient stations before analysis time")
        S = station_order.size
        station_xy = roster[["gx", "gy"]].to_numpy(dtype=np.float32)
        station_index = {str(sid): i for i, sid in enumerate(station_order)}
        Ch = len(self.hist_names)
        station_hist = np.zeros((S, self.past_len, Ch), dtype=np.float32)
        station_hist_mask = np.zeros_like(station_hist)
        for ti, ts in enumerate(past_times):
            fa = self._frame_arrays(ts)
            valid_pairs = [
                (i, station_index[str(sid)])
                for i, sid in enumerate(fa["sids"])
                if str(sid) in station_index
            ]
            if valid_pairs:
                local_idx, global_idx = np.asarray(valid_pairs, dtype=np.int64).T
                station_hist[global_idx, ti] = fa["values"][local_idx, :Ch]
                station_hist_mask[global_idx, ti] = fa["masks"][local_idx, :Ch]
        self.station_ids = station_order
        return {
            "past_grid": torch.from_numpy(past_grid),
            "future_grid": torch.from_numpy(future_grid),
            "station_xy": torch.from_numpy(station_xy),
            "station_hist": torch.from_numpy(station_hist),
            "station_hist_mask": torch.from_numpy(station_hist_mask),
            "station_valid": torch.ones(S, dtype=torch.bool),
            "c_hist": torch.tensor([c_hist], dtype=torch.float32),
            "c_raw": torch.tensor([c_info.get("raw", 1.0)], dtype=torch.float32),
            "n_station": torch.tensor([S], dtype=torch.long),
        }
