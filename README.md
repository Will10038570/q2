# 多攝影機 Tracklet 關聯資料集

## 任務

給定來自四支已同步 CCTV 攝影機的 local tracklets，請將屬於同一個實際道路使用者的 tracklets 進行關聯，並輸出一致的 global identity。

此資料集刻意保留單攝影機 tracking 不完美、distractor tracks、world coordinate 缺失／無效，以及小幅殘餘時間同步誤差等情況。

## 檔案

- `dev/tracklets.csv` — development set 的 tracklets
- `dev/ground_truth.csv` — development set 的 global ID 標註
- `test/tracklets.csv` — hidden test 的 tracklets；不包含標註
- `camera_metadata.json` — 基本 metadata
- `eval.py` — 用於 development set 的跨攝影機 pairwise association evaluator

## Tracklet 欄位

`tracklets.csv` 包含：

- `camera_id`
- `frame_id`
- `timestamp` — 已同步的 global timestamp，單位為秒
- `local_track_id`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `class`
- `confidence`
- `world_x`, `world_y` — 所有攝影機共用的局部笛卡兒座標，單位為公尺
- `world_valid` — `1` 表示 world coordinate 可使用，否則為 `0`

請勿假設每個 local track 都一定能在另一支攝影機中找到對應 track。

## Development 標註

`dev/ground_truth.csv` 會將一部分可靠的 local tracks 對應到 global identity：

`camera_id, local_track_id, global_id`

未列在 ground-truth 檔案中的 tracks，應視為 unlabeled / distractor tracks，而不是自動視為負樣本 identity。

## 評估

提供的 evaluator 會針對有標註的 tracks 計算 pairwise association，且只比較來自不同攝影機的 track pair。

```bash
python eval.py --gt dev/ground_truth.csv --pred prediction.csv
```

### 評估維度

除整體 pairwise F1 外，本專案採用以下拆分視角進一步診斷：

| 維度 | 說明 |
|---|---|
| **整體** | 所有 GT tracks 的 pairwise P / R / F1 |
| **Per camera pair** | A-B、A-C、A-D、B-C、B-D、C-D 分別計算 |
| **Per class** | car / truck / bus 分別計算（dev set GT 中三類均有出現） |
| **Sequential vs Overlapping pairs** | GT pair 有時間重疊 → Overlapping；gap > 0 → Sequential（佔 65%）；分別計算 R，以診斷 temporal gate 設計的合理性 |

---

## Dev Set 資料分析

| 項目 | 數值 |
|---|---|
| 攝影機數 | 4（A / B / C / D） |
| Local tracks 總數 | 617 |
| 標注 global IDs | 92（涵蓋 286 tracks） |
| 時間範圍 | 0 – 211.56 秒 |
| world_valid=0 比例 | 12%（4305 / 35577 frames） |
| frames per track（中位數） | 16（min=1, max=1994） |
| GT tracks：min frames | 6（無 GT track ≤ 5 frames） |
| GT 中出現的 class | car（256）、truck（29）、bus（1）；person / scooter 完全不在 GT 中 |

### GT Pair 時間分布

| 統計量 | 值 |
|---|---|
| GT pairs 總數（跨攝影機） | 308 |
| Overlapping（時間重疊） | 108（35%） |
| Sequential（gap > 0） | 200（65%） |
| Sequential gap 中位數 | 0.64 秒 |
| Sequential gap p90 | 13.28 秒 |
| Sequential gap p95 | 32.18 秒 |
| Sequential gap 最大值 | 69.2 秒 |

**關鍵發現**：大多數 GT pairs 是 sequential（一條 track 結束後另一條才開始），而非同時可見。這反映了車輛穿越路口時依序出現在不同攝影機的真實場景。

---

## 實作架構

### 整體流程

```
tracklets.csv
  → data_loader.py        # 讀取、彙整成 per-tracklet summary
  → analyze_pair_dist.py  # 從 GT calibrate 各項 threshold
  → baseline.py           # distractor filter → gate 判斷 → Union-Find → global_id
  → evaluator.py          # pairwise F1（整體 + per camera-pair + per class + sequential/overlapping）
```

---

## 現行做法

### Phase 1：Distractor Filter

以下條件任一成立即標為 distractor（不參與 association）：

| 條件 | 閾值 | 依據 |
|---|---|---|
| class ∈ {person, scooter} | — | GT 中完全沒有這兩類 |
| frames 數 ≤ n_frames_thresh | 5 | GT min frames = 6 |
| world_valid frames 數 = 0 | — | GT 中無此情況 |
| mean confidence < conf_thresh | 0.45 | 人工設定 |

### Phase 2：Association Gate

對每個跨攝影機 track pair 依序套用以下 gate，全部通過才納入 compat set：

1. **Class gate**：class 不同 → 排除
2. **Temporal gate**：`gap = max(t_start_a, t_start_b) - min(t_end_a, t_end_b)`；`gap > T_max_gap` → 排除（允許 sequential pairs）
3. **Spatial gate（Overlapping，gap ≤ 0）**：重疊時間窗內 per-frame 距離平均值 ≤ D_overlap_thresh；無有效 frame pair 時跳過
4. **Spatial gate（Sequential，gap > 0）**：earlier track 最後一個 world_valid frame 與 later track 第一個 world_valid frame 的距離 ≤ D_seq_thresh；任一 track 無 world_valid frame 時跳過
5. **Speed gate（Sequential only）**：`endpoint_dist / gap > V_max` → 排除

### Threshold Calibration

所有 threshold 由 `analyze_pair_dist.py` 從 GT pairs 自動 calibrate：

| 參數 | 值 | Calibration 依據 |
|---|---|---|
| T_max_gap | 13.28 s | GT sequential gaps p90 |
| V_max | 76.18 m/s | GT sequential required speeds p95 |
| D_overlap（per pair） | 3.23–83.47 m | GT overlapping per-frame distances p90 |
| D_seq（per pair） | 21.14–216.27 m | GT sequential endpoint distances p90 |

### Clustering

通過 gate 的 compat pairs 以 **Union-Find** 合併：對每個 compat pair 直接 union，無額外限制。

### 結果（dev set）

**整體指標**

| Metric | 值 |
|---|---|
| Precision | 0.011 |
| Recall | 0.864 |
| F1 | 0.022 |
| TP | 266 |
| FP | 24,043 |
| FN | 42 |
| Clusters formed | 7 |

**Per Camera Pair**

| Camera Pair | Precision | Recall | F1 |
|---|---|---|---|
| A-B | 0.010 | 0.816 | 0.020 |
| A-C | 0.012 | 0.915 | 0.024 |
| A-D | 0.011 | 0.850 | 0.022 |
| B-C | 0.008 | 0.955 | 0.017 |
| B-D | 0.011 | 0.831 | 0.023 |
| C-D | 0.011 | 0.882 | 0.022 |

**Per Class**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| car | 0.010 | 1.000 | 0.021 |
| truck | 0.097 | 0.824 | 0.173 |
| bus | 0.000 | 0.000 | 0.000 |

**Temporal Split**

| Pair Type | Total Pairs | Recall |
|---|---|---|
| Overlapping | 108 | 0.898 |
| Sequential | 200 | 0.845 |

**分析**：376 條 valid track 被合成 7 個 cluster，幾乎所有 track 都被串進同一個大 cluster。Union-Find 的傳遞性在 compat set FP 率達 93.5%（3537/3782）的情況下，FP pairs 像鏈條一樣把不相關 track 串聯，導致 Precision 崩潰（0.011）。Recall 高（0.864）但無意義，因為幾乎所有 track 都屬於同一個 cluster。

---

## 執行方式

```bash
# 安裝依賴
pip install pandas

# 執行 baseline 並評估
python run_baseline.py

# 單獨分析 GT 距離分布（calibrate thresholds）
python analyze_pair_dist.py
```
