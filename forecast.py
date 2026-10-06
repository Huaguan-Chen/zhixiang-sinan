import argparse
from pathlib import Path
from sinan import core
from sinan.pipeline import STAGES, forecast, load_config, preflight


def main():
    parser = argparse.ArgumentParser(description="zhixiang·sinan inference")
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/inference.yaml")
    )
    parser.add_argument("--analysis-time", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg, base_dir = load_config(args.config)
    analysis_time = core.parse_time(args.analysis_time)
    preflight(cfg, base_dir, args.stage)
    if args.check:
        print("Configuration and input paths are available")
        return
    output = forecast(
        cfg,
        base_dir,
        analysis_time,
        core.ensure_device(args.device),
        args.output_root,
        args.stage,
    )
    print(f"Forecast saved to {output}")


if __name__ == "__main__":
    main()
