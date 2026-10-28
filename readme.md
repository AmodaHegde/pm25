# Physics Informed Neural Network for pollutant-atmospheric modelling

This is the Master's project for understanding, modelling and predicting air pollution in Delhi using a Physics Informed Neural Network.
Baseline - LSTM, Transformer.

## Project contents

- `data/bronze/` and `data/silver/` - raw and processed datasets. Some large datasets are tracked with DVC pointer files (`*.csv.dvc`).
- `src/data/` - scripts to retrieve Open-Meteo air-quality and weather data and to merge station air and weather data.
- `src/featengg/feature_engineering.py` - derives cyclical time, wind, domain, lag, and rolling-window features for the PM2.5 dataset.
- `src/eda/` - exploratory analyses and generated summaries/plots in `results/`.
- `src/models/` - LSTM, Transformer, and physics-informed neural network (PINN) experiments.
- `utils/tune_lstm.py` - Optuna hyperparameter tuning for the LSTM.
- `params.yaml` - model and training configuration.
- `dvc.yaml` - DVC stage definition for the PINN cross-validation experiment.

## Setup

Use Python 3.10 or newer, then install the listed dependencies from the project root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch can require a platform-specific install to enable GPU support; use the appropriate PyTorch installation for your system if needed.

## Data

The checked-in `data/` directory contains the inputs used during development. DVC-managed files may need to be fetched from the configured remote:

```bash
dvc pull
```

`src/data/get.py` retrieves hourly air-quality and archive weather observations for Delhi from Open-Meteo and writes a bronze CSV. The latitude, longitude, date range, and output path are defined in that script. `src/featengg/feature_engineering.py` reads the bronze PM2.5 CSV and writes the engineered silver dataset.

Those scripts currently contain absolute paths rooted at `D:/miniproject`; update those paths to your local project directory before running them elsewhere. Data collection also requires network access and availability of the Open-Meteo API.

## Running experiments

Run commands from the repository root because the model scripts load `params.yaml` and datasets using paths relative to the root.

```bash
python src/models/lstm.py
python src/models/transformer.py
python utils/tune_lstm.py --n-trials 30
```

The LSTM and Transformer log experiments to MLflow. Start the MLflow UI from the root to inspect local runs:

```bash
mlflow ui
```

The PINN's DVC stage is declared in `dvc.yaml`; run it with `dvc repro` after its input dataset is available. The model scripts and DVC stage may require configuration adjustments for a different machine or dataset layout.

## Configuration and outputs

Edit `params.yaml` to change data paths, lookback windows, model dimensions, training settings, and MLflow experiment names. Training scripts save model/scaler artifacts under `models/` as configured. EDA outputs are written to `results/`; MLflow tracking data is stored locally by default.

The scripts are research/experimentation code and are not yet packaged as a command-line application. Review input paths and dataset columns before launching a run.
