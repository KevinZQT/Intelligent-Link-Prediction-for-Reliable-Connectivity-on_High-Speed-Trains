# Short‑Term Downlink Throughput Prediction on Trains

> **Thesis companion repository.** This repo contains data collection utilities, feature engineering, and model pipelines to perform **t → t+1** one‑step prediction of cellular **downlink throughput (Mbps)** at 1 Hz in a real commuter‑rail setting.

The work uses real‑world measurements collected on the Dublin DART line with multiple USB modems (different operators) and phone GPS. We engineer strictly‑causal features and evaluate several models under a leakage‑safe time‑ordered split.

---

## ✨ Highlights

- **Real data collection pipeline:** Linux SBC + ModemManager (`mmcli`) for radio KPIs; continuous reverse‑direction `iperf3` downlink to measure per‑second throughput; 1 Hz GPS from a phone, with UTC alignment.
- **Leakage‑safe evaluation:** **80/20 chronological split** with a **25 s embargo** before the test window to cover the longest lag (20 s) + safety margin.
- **Causal features:** Current values, short histories (throughput lags up to **20 s**, radio KPI lags up to **3 s**), 5‑second rolling statistics (mean/std/min/max), momentum/differences, volatility, and simple state flags (e.g., handover).
- **Model zoo:** Persistence & **AR(1)** baselines; classical ML (**Random Forest**), **ANN/MLP**; sequence model **TCN**; and a **TCN+DQN** hybrid variant.
- **Key finding (from the thesis):** On held‑out tests for two representative modems, **Random Forest** achieves **R² ≈ 0.53/0.37** with **MAE ≈ 17.1/19.5 Mbps**, clearly outperforming linear AR(1) (negative R², MAE ≈ 35.6/33.2 Mbps).

---

## 📁 Repository Layout

```
.
├─ data_collect.py          # Field collection: mmcli + iperf3 (1 Hz throughput & KPIs) → raw CSV
├─ data_merge.ipynb         # Align & clean raw modem/GPS/iperf3 streams to a single 1 Hz timeline
├─ persistent_ar1.py        # Baselines: Persistence, AR(1); includes a simple RF check; unified metrics
├─ rf.py                    # Random Forest (residual target y_{t+1} - y_t) + feature selection + scaling
├─ ann.py                   # MLP/ANN with the same engineered features & evaluation protocol
├─ tcn.py                   # Temporal Convolutional Network (causal/dilated convs, early stopping)
├─ tcn+dqn.ipynb            # Hybrid TCN + DQN experiment (policy/threshold refinement on TCN output)
└─ thesis.pdf               # The dissertation (rename of 9c5cc53e-fffc-44cb-983a-527f2f6d3ad8.pdf)
```

> Tip: you can rename the provided thesis PDF to `thesis.pdf` in the repository.

---

## 🧰 Environment

- **Python** 3.10+ (recommended)
- **Core:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- **Deep learning:** `tensorflow` (for ANN/TCN)
- **Notebooks:** `jupyter` (for the `*.ipynb` files)
- **Collection (optional, Linux):** `modemmanager` (`mmcli`), `iperf3`

Quick install:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib jupyter
# optional for collection
# sudo apt-get install modemmanager iperf3
```

---

## 📦 Data Expectations

Most training scripts expect a **processed** CSV at the repo root, named **`data_filled.csv`**, with at least:

- `timestamp` (UTC), `modem_id`, `downlink_mbps`
- If available: `rssi, rsrp, rsrq, snr, speed_gps, cell_id, ...`

The collection script `data_collect.py` writes raw CSVs (default `modem_downlink_only.csv`) from `mmcli` and `iperf3`. Use `data_merge.ipynb` to **align** and **clean** the raw streams into the 1 Hz `data_filled.csv` used by the models.

---

## 🚀 Quick Start (with processed data)

1. **Place** your processed **`data_filled.csv`** in the repository root.
2. **Run** any of the following (each trains & evaluates **per‑modem**, using the same split/embargo and metrics):

```bash
# Baselines: Persistence + AR(1) + a simple RF sanity check
python persistent_ar1.py

# Random Forest (thesis best overall)
python rf.py

# MLP / ANN (same engineered features and protocol)
python ann.py

# Temporal Convolutional Network (causal/dilated)
python tcn.py
```

Each script prints a per‑modem summary (R², MAE, MAPE vs current, and useful distributional stats).

---

## 🧪 Reproducibility & Evaluation Protocol

- **Objective:** one‑step forecast **`y_{t+1}`** (next‑second downlink Mbps). Many scripts learn the **residual** `y_{t+1} - y_t` and add back `y_t` at inference.
- **Split:** time‑ordered; last **20%** as **test**, with a **25 s embargo** immediately before test to prevent look‑ahead via lagged features.
- **Adjacency constraint:** evaluate only pairs with **Δt ≤ 1.1 s** (tightly 1 Hz).
- **Metrics:** MAE (Mbps), R², and percentage errors w.r.t. current rate (MAPE‑style; treat near‑zero throughput with caution).

These rules are encoded in the training scripts; no extra config needed for a standard run.

---

## ⚙️ Configuration (key defaults)

- **Feature lags:** throughput lags **1..20 s**, radio‑KPI lags **1..3 s**.
- **Rolling stats:** window **5 s** (mean/std/min/max).
- **Feature selection:** `SelectKBest(f_regression)` up to **30** features; `RobustScaler` for stability.
- **Split & safety:** `test_ratio=0.2`, **25 s embargo**, adjacency **≤ 1.1 s**.
- **Training (ANN/TCN):** early stopping (`patience=15`), `ReduceLROnPlateau`, small min‑LR, light hyper‑param tuning toggles.
- **Percentage error floor:** 0.1 Mbps to avoid division blow‑ups at ~0 throughput.

Script‑specific notes:

- `ann.py` reads `CFG.input_csv` (default **`data_filled.csv`**).
- `rf.py` and `persistent_ar1.py` read **`data_filled.csv`** directly.
- `tcn.py` accepts `--data PATH` (defaults to a CSV path); it builds features internally before training.

---

## 📊 Key Results (from the thesis)

On two representative modems in the held‑out test split:

| Model             | R² (M1)  | MAE (Mbps, M1) | R² (M2)  | MAE (Mbps, M2) |
| ----------------- | :------: | :------------: | :------: | :------------: |
| Persistence       |    —     |       —        |    —     |       —        |
| AR(1)             |  -0.49   |      35.6      |  -0.50   |      33.2      |
| **Random Forest** | **0.53** |    **17.1**    | **0.37** |    **19.5**    |
| ANN / MLP         |  ~0.51   |     ~18–20     |  ~0.21   |     ~23–24     |
| TCN               |  ~0.23   |     ~24–25     |  ~0.10   |     ~25–26     |
| TCN + DQN         |  ~0.19   |     ~25–30     |  ~0.11   |      ~30       |

> RF consistently achieves the best accuracy–robustness trade‑off in this small‑sample, rapidly varying setting. See **`thesis.pdf`** for complete tables (incl. MAPE and distributional quantiles).

---

## 🧱 (Optional) Data Collection Workflow

1. **Hardware & setup:** Linux SBC + 3× USB cellular modems (multiple operators).
2. **KPIs:** use `mmcli` to poll RSRP/RSRQ/RSSI/SNR, EARFCN, RAT, MCC/MNC, TAC, Cell ID, etc., at 1 Hz.
3. **Throughput:** run `iperf3` in **reverse** (server → client) downlink mode; parse per‑second rates.
4. **GPS:** 1 Hz phone GPS; ensure UTC synchronization with the SBC.
5. **Merge:** align modem/GPS/iperf3 streams to a single 1 Hz axis; carefully handle gaps/handovers (no cross‑segment forward‑fill); clamp outliers (e.g., [0, 200] Mbps).

`data_collect.py` shows a minimal reference implementation (concurrency for `iperf3`, `mmcli` parsing, CSV writing).

---

## 🔁 Re‑running the Thesis Experiments

- Use `data_merge.ipynb` to reproduce the **processed** dataset used in the thesis (`data_filled.csv`).
- Then run `rf.py`, `ann.py`, and `tcn.py` as above.
- For the hybrid experiment, open `tcn+dqn.ipynb` and follow the notebook cells.

---

## 📄 License

Choose a license that fits your needs (e.g., **MIT** or **Apache‑2.0**). Add a `LICENSE` file at the repo root.

---

## 📝 Citation

If this repository or the associated thesis is useful to your research, please cite it. Replace placeholders below with your final bibliographic information.

```
@thesis{<your-key>,
  title   = {Short-Term Downlink Throughput Prediction on Trains},
  author  = {<Your Name>},
  school  = {<Your University/Department>},
  year    = {2025},
  note    = {Code and data processing pipelines available at <your GitHub URL>}
}
```

---

## 🙏 Acknowledgements

Thanks to everyone who helped with data collection and to the network operators whose infrastructure made this study possible.
