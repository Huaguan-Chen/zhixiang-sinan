from __future__ import annotations
import gc
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import core
from .radar.inference import infer_radar
from .station import data as station_data
from .station import model_a, model_b

STAGES = ("radar", "observations", "station", "grid", "accumulation")


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = core.read_yaml(path)
    for section, keys in {
        "radar": ("raw_root",),
        "station": ("raw_root",),
        "stage1": (
            "base_checkpoint",
            "diffusion_new_checkpoint",
            "diffusion_0_checkpoint",
        ),
        "model_a": ("config", "checkpoint"),
        "model_b": ("config", "checkpoint"),
        "fixed_stations": ("catalog",),
    }.items():
        for key in keys:
            if cfg.get(section, {}).get(key):
                cfg[section][key] = str(
                    core.resolve_path(cfg[section][key], path.parent)
                )
    return (cfg, path.parent)


def preflight(cfg, base_dir, stage="all"):
    required = []
    stages = STAGES if stage == "all" else (stage,)
    if "radar" in stages:
        required.append(Path(cfg["radar"]["raw_root"]))
        required.extend(
            (
                Path(cfg["stage1"][key])
                for key in (
                    "base_checkpoint",
                    "diffusion_new_checkpoint",
                    "diffusion_0_checkpoint",
                )
            )
        )
    if "observations" in stages:
        required.append(Path(cfg["station"]["raw_root"]))
    if "station" in stages:
        for key in ("model_a", "model_b"):
            required.extend((Path(cfg[key][name]) for name in ("config", "checkpoint")))
        if cfg.get("fixed_stations", {}).get("enabled", False):
            required.append(Path(cfg["fixed_stations"]["catalog"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if int(cfg["radar"].get("interval_minutes", 6)) != 6:
        raise ValueError("These checkpoints require a 6-minute interval")
    if (
        int(cfg["radar"].get("past_len", 10)) != 10
        or int(cfg["radar"].get("future_len", 30)) != 30
    ):
        raise ValueError("These checkpoints require 10 input and 30 forecast frames")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_radar(cfg, paths, analysis_time, device):
    raw_window = core.build_raw_radar_window(cfg, analysis_time, paths.raw_window_json)
    scfg = cfg["stage1"]
    infer_radar(
        root_dir=str(paths.case_dir),
        json_file=paths.raw_window_json.name,
        save_root=str(paths.stage1_root),
        ckpt_path=scfg["base_checkpoint"],
        diff_new_ckpt=scfg["diffusion_new_checkpoint"],
        diff_0_ckpt=scfg["diffusion_0_checkpoint"],
        batch_size=int(scfg.get("batch_size", 1)),
        num_workers=int(scfg.get("num_workers", 2)),
        dt_minutes=6,
        anchor_time=core.parse_time(
            scfg.get("anchor_time", "20230101_000000")
        ).to_pydatetime(),
        lat_range=tuple(cfg["domain"]["lat_range"]),
        lon_range=tuple(cfg["domain"]["lon_range"]),
        normalize_div=float(cfg["radar"].get("raw_normalize_div", 800.0)),
        crop_last_dim_to=int(cfg["domain"]["w"]),
        diff_threshold=float(scfg.get("diff_threshold", 0.1)),
        diff_min_size=int(scfg.get("diff_min_size", 4)),
        save_denorm_out_all=True,
        device_name=str(device),
    )
    window = core.preprocess_radar_to_row0south(cfg, raw_window, paths)
    orientation = core.assert_orientation_contract(cfg, window)
    velocity_dir = paths.stage1_root / paths.raw_window_json.stem / "veolicity"
    motion_paths = [
        velocity_dir / f'{record['time']}_veolicity.npz' for record in window["future"]
    ]
    if all((path.exists() for path in motion_paths)):
        for record, path in zip(window["future"], motion_paths):
            velocity = torch.as_tensor(core.load_np_any(path), dtype=torch.float32)[
                None, None
            ]
            if cfg["radar"].get("apply_raw_radar_y_flip", True):
                velocity = torch.flip(velocity, dims=[-2])
            motion = core._convert_velocity_grid_per_6min_to_mps(
                velocity, cfg["domain"]["lat_range"], grid_deg=0.01, dt_minutes=6
            )
            destination = (
                paths.output_dir / "radar_motion" / f'{record['time']}_motion_mps.npz'
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                motion_mps=motion[0, 0].numpy(),
                orientation=np.asarray("row0south"),
            )
    summary = {
        "analysis_time": core.format_time(analysis_time),
        "station_window_json": str(paths.station_window_json),
        "orientation": orientation,
    }
    core.write_json(paths.output_dir / "step1_radar_forecast_summary.json", summary)
    clear_memory()


def run_observations(cfg, paths, analysis_time):
    raw_window = json.loads(paths.raw_window_json.read_text(encoding="utf-8"))[0]
    station_frame = core.process_raw_station_data(cfg, analysis_time, raw_window, paths)
    core.write_json(
        paths.output_dir / "step2_station_processing_summary.json",
        {
            "analysis_time": core.format_time(analysis_time),
            "rows": len(station_frame),
            "station_6min_root": str(paths.station_6min_root),
        },
    )


def model_config(cfg, key, paths):
    result = deepcopy(core.read_yaml(Path(cfg[key]["config"])))
    result["paths"] = {
        "root_dir": str(paths.case_dir),
        "window_json": str(paths.station_window_json),
        "aws_6min_root": str(paths.station_6min_root),
    }
    result["data"]["flip_pred_y"] = False
    result["persistent_frame_cache"] = {"enabled": False}
    if result["domain"].get("north_to_south_y", True):
        raise ValueError("Station models require row0south coordinates")
    return result


def predict_station(module, cfg, sample, checkpoint, device):
    model = module.TemporalUNetStationUnionModel(
        cfg,
        past_ch=int(sample["past_grid"].shape[1]),
        future_ch=int(sample["future_grid"].shape[1]),
        hist_dim=int(sample["station_hist"].shape[-1]),
        num_targets=len(station_data.VARIABLE_SPECS),
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key
        in (
            "past_grid",
            "future_grid",
            "station_xy",
            "station_hist",
            "station_hist_mask",
            "station_valid",
        )
    }
    amp = cfg.get("inference", {})
    dtype = torch.bfloat16 if amp.get("amp_dtype", "bf16") == "bf16" else torch.float16
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda" and bool(amp.get("amp", True)),
            dtype=dtype,
        ),
    ):
        output = model(batch)["preds"][0].float().cpu().numpy()
    del batch, model, state
    clear_memory()
    return output


def run_station(cfg, base_dir, paths, analysis_time, device):
    cfg_a = model_config(cfg, "model_a", paths)
    cfg_b = model_config(cfg, "model_b", paths)
    ds = station_data.StationForecastDataset(cfg_a)
    sample = ds[0]
    station_ids = ds.station_ids
    core.validate_real_station_sample(sample, station_ids)
    active = np.ones(station_ids.size, dtype=bool)
    catalog = core.load_fixed_station_catalog(cfg, base_dir)
    if catalog is not None:
        sample, station_ids, active = core.reindex_station_sample_to_catalog(
            sample, station_ids, catalog, cfg
        )
    core.validate_real_station_sample(sample, station_ids)
    if bool(cfg_a["data"].get("compact_future_rain_only", False)) != bool(
        cfg_b["data"].get("compact_future_rain_only", False)
    ):
        raise ValueError("Model A/B must use the same future feature layout")
    pred_a = predict_station(
        model_a, cfg_a, sample, cfg["model_a"]["checkpoint"], device
    )
    pred_b = predict_station(
        model_b, cfg_b, sample, cfg["model_b"]["checkpoint"], device
    )
    variables = ds.variable_specs
    physical_a = np.stack(
        [core.to_physical(pred_a, variables, i) for i in range(len(variables))], axis=-1
    )
    physical_b = np.stack(
        [core.to_physical(pred_b, variables, i) for i in range(len(variables))], axis=-1
    )
    rain_index = core.spec_index(variables, "pre6")
    a_rain = np.maximum(physical_a[..., rain_index], 0.0)
    b_rain = np.maximum(physical_b[..., rain_index], 0.0)
    xy = sample["station_xy"].numpy()
    lon_min, lon_max = cfg["domain"]["lon_range"]
    lat_min, lat_max = cfg["domain"]["lat_range"]
    lon = lon_min + (xy[:, 0] + 1.0) * 0.5 * (lon_max - lon_min)
    lat = lat_min + (xy[:, 1] + 1.0) * 0.5 * (lat_max - lat_min)
    window = json.loads(paths.station_window_json.read_text(encoding="utf-8"))[0]
    fields = core.radar_forecast_fields(station_data, cfg_a, ds, window, sample)
    np.savez_compressed(
        paths.output_dir / "station_model_forecast_raw.npz",
        station_ids=station_ids.astype(str),
        station_xy=xy.astype(np.float32),
        station_lon=lon.astype(np.float32),
        station_lat=lat.astype(np.float32),
        station_active_mask=active,
        station_history_valid_count=sample["station_hist_mask"]
        .sum(dim=(1, 2))
        .numpy()
        .astype(np.int32),
        future_times=np.asarray([record["time"] for record in window["future"]]),
        variable_names=np.asarray([spec.name for spec in variables]),
        model_a_variables=physical_a,
        model_b_variables=physical_b,
        model_a_pre6_mm=a_rain,
        model_b_pre6_mm=b_rain,
        model_a_temperature=physical_a[..., core.spec_index(variables, "tem")],
        model_a_wind_speed=np.maximum(
            physical_a[..., core.spec_index(variables, "s_2min")], 0.0
        ),
        **fields,
    )
    summary = {
        "analysis_time": core.format_time(analysis_time),
        "station_count": int(station_ids.size),
        "active_station_count": int(active.sum()),
        "model_a": cfg["model_a"]["name"],
        "model_b": cfg["model_b"]["name"],
    }
    core.write_json(paths.output_dir / "step3_station_forecast_summary.json", summary)
    del sample, ds
    clear_memory()


def run_grid(cfg, paths, analysis_time, device):
    with np.load(
        paths.output_dir / "station_model_forecast_raw.npz", allow_pickle=False
    ) as source:
        data = {name: source[name] for name in source.files}
    backgrounds = data["corrected_pred_y0_rain_mm"]
    active = (
        data["station_active_mask"]
        if cfg.get("fixed_stations", {}).get("assimilation_active_only", True)
        else np.ones(data["station_ids"].size, dtype=bool)
    )
    if int(active.sum()) < int(cfg["assimilation"].get("min_stations", 5)):
        raise RuntimeError("Insufficient active stations for assimilation")
    a = data["model_a_pre6_mm"]
    b = data["model_b_pre6_mm"]
    xy = data["station_xy"]
    fused = np.stack(
        [
            core.fuse_station_pre6(bg, xy, a[i], b[i], cfg["assimilation"], device)
            for i, bg in enumerate(backgrounds)
        ]
    )
    grids = np.stack(
        [
            core.assimilate_ab_to_grid(
                bg, xy[active], a[i, active], b[i, active], cfg["assimilation"], device
            )
            for i, bg in enumerate(backgrounds)
        ]
    )
    fused[fused < 0.01] = 0.0
    grids[grids < 0.01] = 0.0
    rows = []
    for lead, valid_time in enumerate(data["future_times"].astype(str)):
        for index, station_id in enumerate(data["station_ids"].astype(str)):
            row = {
                "analysis_time": core.format_time(analysis_time),
                "valid_time": valid_time,
                "lead": lead,
                "station_id": station_id,
                "lon": float(data["station_lon"][index]),
                "lat": float(data["station_lat"][index]),
                "active_for_assimilation": bool(active[index]),
                "history_valid_count": int(data["station_history_valid_count"][index]),
                "model_a_pre6_mm": float(a[lead, index]),
                "model_b_pre6_mm": float(b[lead, index]),
                "fused_pre6_mm": float(fused[lead, index]),
                "model_a_temperature": float(data["model_a_temperature"][lead, index]),
                "model_a_wind_speed": float(data["model_a_wind_speed"][lead, index]),
            }
            row.update(
                {
                    f"model_a_{name}": float(
                        data["model_a_variables"][lead, index, variable]
                    )
                    for variable, name in enumerate(data["variable_names"].astype(str))
                }
            )
            rows.append(row)
    core.write_csv_rows(paths.output_dir / "station_forecast.csv", rows)
    fields = {
        name: data[name]
        for name in data
        if name.endswith("_norm") or name.endswith("_rain_mm")
    }
    np.savez_compressed(
        paths.output_dir / "radar_grid_forecast_and_assimilation.npz",
        **fields,
        corrected_radar_background_mm=backgrounds,
        assimilated_model_ab_mm=grids,
        future_times=data["future_times"],
        orientation=np.asarray("row0south"),
    )
    core.write_json(
        paths.output_dir / "run_summary.json",
        {
            "analysis_time": core.format_time(analysis_time),
            "station_count": int(a.shape[1]),
            "assimilation_station_count": int(active.sum()),
            "future_times": data["future_times"].tolist(),
            "station_forecast": str(paths.output_dir / "station_forecast.csv"),
            "grid_forecast": str(
                paths.output_dir / "radar_grid_forecast_and_assimilation.npz"
            ),
        },
    )


def rolling_accumulation(values, window):
    values = np.asarray(values, dtype=np.float32)
    result = np.full_like(values, np.nan)
    for index in range(window - 1, values.shape[0]):
        result[index] = np.sum(values[index - window + 1 : index + 1], axis=0)
    return result


def run_accumulation(paths):
    df = pd.read_csv(
        paths.output_dir / "station_forecast.csv", dtype={"station_id": str}
    )
    ordered = df.sort_values(["lead", "station_id"]).reset_index(drop=True)
    station_ids = ordered.loc[
        ordered["lead"] == ordered["lead"].min(), "station_id"
    ].to_numpy(dtype=str)
    leads = np.sort(ordered["lead"].unique())
    for source, prefix in [
        ("model_a_pre6_mm", "model_a"),
        ("model_b_pre6_mm", "model_b"),
        ("fused_pre6_mm", "fused"),
    ]:
        values = (
            ordered.pivot(index="lead", columns="station_id", values=source)
            .reindex(index=leads, columns=station_ids)
            .to_numpy(dtype=np.float32)
        )
        for window, hours in [(10, 1), (30, 3)]:
            ordered[f"{prefix}_{hours}h_mm"] = rolling_accumulation(
                values, window
            ).reshape(-1)
    ordered.to_csv(paths.output_dir / "station_forecast_with_accum.csv", index=False)
    with np.load(
        paths.output_dir / "radar_grid_forecast_and_assimilation.npz",
        allow_pickle=False,
    ) as source:
        result = {
            "future_times": source["future_times"],
            "orientation": source["orientation"],
        }
        for name, prefix in [
            ("pred_y0_rain_mm", "radar"),
            ("assimilated_model_ab_mm", "assimilated"),
        ]:
            for window, hours in [(10, 1), (30, 3)]:
                result[f"{prefix}_{hours}h_mm"] = rolling_accumulation(
                    source[name], window
                )
        np.savez_compressed(paths.output_dir / "grid_accumulated_1h_3h.npz", **result)


def forecast(cfg, base_dir, analysis_time, device, output_root="", stage="all"):
    paths = core.runtime_paths(cfg, base_dir, analysis_time, output_root)
    paths.case_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if stage == "all" else (stage,)
    for current in stages:
        print(f"[{current}] {core.format_time(analysis_time)}", flush=True)
        if current == "radar":
            run_radar(cfg, paths, analysis_time, device)
        elif current == "observations":
            run_observations(cfg, paths, analysis_time)
        elif current == "station":
            run_station(cfg, base_dir, paths, analysis_time, device)
        elif current == "grid":
            run_grid(cfg, paths, analysis_time, device)
        else:
            run_accumulation(paths)
    return paths.output_dir
