# zhixiang·sinan

仅包含推理代码：原始三维雷达 → 雷达预报 → 站点预报 → 格点降水同化 → 1/3 小时累积降水。每次输入 10 帧历史雷达，输出 30 个时效，间隔 6 分钟，预报范围 3 小时。

```bash
pip install -r requirements.txt
python forecast.py --analysis-time 20260629_150000 --device cuda
```

运行前修改 `configs/inference.yaml` 的数据与权重路径。相对路径以配置文件所在目录为基准。权重需自行放置：

```text
weights/radar_base.pth
weights/radar_diffusion_new.pth
weights/radar_diffusion_zero.pth
weights/station_a.pt
weights/station_b.pt
```

前三项对应原来的 `epoch_180_MPPN_brownian_final_V8.pth`、`diffusion_epoch_2413_new.pth`、`diffusion_epoch_2080.pth`。站点 A 使用 compact lowres UNet residual 模型的 `best.pt`，站点 B 使用 direct clamp 模型的 `best.pt`。保持配置中的模型参数不变。

雷达数据使用 `{YYYYMMDD_HHMMSS}_data.npy` 和对应 `_mask.npy`，保留原始三维数组。站点数据默认使用 `data/stations/YYYYMMDD/cimissYYYYMMDDHHMM.txt`，支持配置其他文件名模式；需包含站号、经纬度、时间和气象观测字段。推理只读取起报时刻及之前的观测。

固定站点预报需提供 `data/fixed_station_catalog.csv`，字段为 `Station_Id_C,Lat,Lon`；使用动态站点时将 `fixed_stations.enabled` 设为 `false`。模型 A/B 的结构与归一化参数分别保存在 `configs/model_a.yaml` 和 `configs/model_b.yaml`，需与现有权重匹配。

结果写入 `results/起报时间/`：`station_forecast.csv` 为站点降水、温度、风速及其他模型变量，`radar_grid_forecast_and_assimilation.npz` 为雷达和格点降水，`station_forecast_with_accum.csv` 与 `grid_accumulated_1h_3h.npz` 为累积降水。累积量仅使用本次预报，前 9 个时效的 1 小时累积和前 29 个时效的 3 小时累积为空。格点统一为第 0 行在南侧。

站点变量还包含 1 小时降水、瞬时/2 分钟/10 分钟风分量和风速、露点、相对湿度、水汽压及气压；原始模型变量保存在 `station_model_forecast_raw.npz`。雷达运动速度另存于 `radar_motion/`。

检查文件路径：`python forecast.py --analysis-time 20260629_150000 --check`。单独运行某一步可加 `--stage radar|observations|station|grid|accumulation`，后续步骤需先有前序步骤的产物。CUDA 用于业务推理，也支持 `--device cpu`。

仓库不包含权重、观测数据、训练代码或代码注释。
