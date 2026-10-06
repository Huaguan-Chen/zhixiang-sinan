# zhixiang·sinan

Radar nowcasting, gridded precipitation, and station weather forecasts for the next 3 hours at 6-minute intervals.

## Usage

Configure the data and model paths in `configs/inference.yaml`, then run:

```bash
pip install -r requirements.txt
python forecast.py --analysis-time 20260629_150000 --device cuda
```

Forecasts are saved to `results/<analysis_time>/`.
