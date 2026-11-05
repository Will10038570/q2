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
  → baseline.py           # distractor filter → gate 判斷 → Union-Find → global_id
  → evaluator.py          # pairwise F1（整體 + per camera-pair + per class + sequential/overlapping）

（Baseline 3，尚未實作）
  → feature_extractor.py  # crop → Re-ID model → per-tracklet appearance embedding
  → affinity_builder.py   # temporal + spatial + Re-ID cosine → affinity matrix
  → matcher.py            # greedy threshold + union-find → global_id
```

---

## Baseline 1 — 原始幾何設計

### 設計

**Distractor Filter（Phase 1）**

以下條件任一成立即標為 distractor（不參與 association）：

| 條件 | 閾值 | 依據 |
|---|---|---|
| frames 數 ≤ n_frames_thresh | 5 | GT min frames = 6 |
| world_valid frames 數 = 0 | — | GT 中無此情況 |
| mean confidence < conf_thresh | 0.45 | 人工設定 |

**Association Gate（Phase 2）**

對每個跨攝影機 track pair 依序套用三道 gate，全部通過才建立關聯：

1. **Class gate（hard）**：class 不同 → 排除
2. **Temporal gate（strict overlap）**：無時間重疊（`min(t_end_a, t_end_b) ≤ max(t_start_a, t_start_b)`）→ 排除
3. **Spatial gate（overlapping pairs）**：在重疊時間窗內，對每個 world_valid frame 找最近的對應 frame（max_dt=1.0s），計算平均 euclidean 距離；超過 D_thresh → 排除。若無有效 frame pair，跳過此 gate。

**Clustering**

通過 gate 的 pairs 以 Clique-based clustering（complete linkage）合併：只有當兩個 cluster 內所有跨 cluster pairs 均通過 gate，才允許合併。

### 核心限制

| 限制 | 說明 |
|---|---|
| Sequential pairs 完全繞過 spatial gate | Temporal gate 為 strict overlap，sequential pairs 在 temporal gate 失效後 spatial gate 也因無重疊 frame 而被跳過，造成大量 FP |
| Person / scooter 未過濾 | GT 中完全沒有這兩類，但它們仍參與 association，貢獻 FP |
| Threshold 人工設定 | D_thresh 未從 GT 分布 calibrate |

### 結果（dev set）

**D_thresh = 20m（T_thresh = 30s 寬鬆版）**

| Metric | 值 |
|---|---|
| Precision | 0.011 |
| Recall | 0.873 |
| F1 | 0.022 |
| TP | 269 |
| FP | 24,199 |
| FN | 39 |

| Camera Pair | Precision | Recall | F1 |
|---|---|---|---|
| A-B | 0.010 | 0.816 | 0.020 |
| A-C | 0.012 | 0.915 | 0.024 |
| A-D | 0.011 | 0.875 | 0.022 |
| B-C | 0.008 | 0.955 | 0.017 |
| B-D | 0.012 | 0.847 | 0.023 |
| C-D | 0.011 | 0.882 | 0.022 |

**分析**：Recall 高（0.87）代表真正配對幾乎都被找到（大多數 sequential pairs 通過了寬鬆的 temporal gate，並因 spatial gate 被跳過而進入 clustering）。Precision 極低（0.011）的根本原因是 sequential pairs 完全缺乏 spatial 約束，導致大量不相關 tracks 被錯誤關聯。

---

## Baseline 2 — Sequential-Aware 幾何設計

### 設計目標

修正 Baseline 1 的三個根本問題：
1. Sequential pairs 缺乏 spatial gate
2. Person / scooter 參與 association 增加 FP
3. Threshold 未從 GT 分布 calibrate

### 改動項目

**Distractor Filter（Phase 1）新增 class gate**

| 條件 | 依據 |
|---|---|
| class ∈ {person, scooter} | GT 中完全沒有這兩類；非 GT tracks 中有 person 16 條、scooter 34 條 |

**Temporal gate 改為 max-gap gate**

原本：有時間重疊才通過（strict overlap）

改為：gap 超過 T_max_gap 才排除

```
gap = max(t_start_a, t_start_b) - min(t_end_a, t_end_b)
若 gap > T_max_gap → 排除
```

| 參數 | GT calibration 依據 |
|---|---|
| T_max_gap（s） | GT sequential gaps 的某百分位數（p90 = 13.28s，p95 = 32.18s）；以 dev set 最佳化 |

**Spatial gate 拆分為兩類**

| Pair 類型 | 判斷方式 | Gate 設計 |
|---|---|---|
| Overlapping（gap ≤ 0） | 有共同時間窗 | 同 Baseline 1：重疊時間窗內 per-frame 距離平均值 ≤ D_overlap_thresh |
| Sequential（gap > 0） | 無共同時間窗 | 新增：earlier track 最後一個 world_valid frame 與 later track 第一個 world_valid frame 的 euclidean 距離 ≤ D_seq_thresh |

**Speed consistency gate（僅 sequential pairs）**

排除物理上不可能的配對：

```
required_speed = endpoint_distance / gap
若 required_speed > V_max → 排除
```

| 參數 | 說明 |
|---|---|
| V_max（m/s） | 以車輛合理最高速估計；從 GT calibrate 或設定合理上限（如 25 m/s ≈ 90 km/h） |

**所有 spatial threshold 從 GT calibrate（per camera pair）**

使用 `analyze_pair_dist.py` 在 GT pairs 上分別估計每個 camera pair 的 D_overlap_thresh 和 D_seq_thresh。

**Clustering**

與 Baseline 1 相同（Clique-based，complete linkage）。

### 新增評估維度（evaluator 擴充）

| 維度 | 實作方式 |
|---|---|
| Per class | 在 GT 中依 class 分組，分別計算 P / R / F1 |
| Sequential vs Overlapping | 依 GT pair 是否有時間重疊分組，分別計算 Recall（Precision 在 pair level 無法直接拆分） |

### 結果（dev set）

**Calibrated thresholds（p90 for distances, p95 for V_max）**

| 參數 | 值 |
|---|---|
| T_max_gap | 13.28 s |
| V_max | 76.18 m/s |
| D_overlap A-B | 76.66 m |
| D_overlap A-C | 3.23 m |
| D_overlap A-D | 3.62 m |
| D_overlap B-C | 69.10 m |
| D_overlap B-D | 11.85 m |
| D_overlap C-D | 83.47 m |
| D_seq A-B | 149.3 m |
| D_seq A-C | 28.21 m |
| D_seq A-D | 22.34 m |
| D_seq B-C | 21.14 m |
| D_seq B-D | 29.36 m |
| D_seq C-D | 216.27 m |

**整體指標**

| Metric | 值 |
|---|---|
| Precision | 0.398 |
| Recall | 0.341 |
| F1 | 0.367 |
| TP | 105 |
| FP | 159 |
| FN | 203 |

| Camera Pair | Precision | Recall | F1 |
|---|---|---|---|
| A-B | 0.652 | 0.612 | 0.632 |
| A-C | 0.488 | 0.426 | 0.455 |
| A-D | 0.269 | 0.225 | 0.245 |
| B-C | 0.286 | 0.273 | 0.279 |
| B-D | 0.326 | 0.254 | 0.286 |
| C-D | 0.372 | 0.314 | 0.340 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| car | 0.382 | 0.373 | 0.378 |
| truck | 0.611 | 0.647 | 0.629 |
| bus | 0.000 | 0.000 | 0.000 |

| Pair Type | Total Pairs | Recall |
|---|---|---|
| Overlapping | 108 | 0.509 |
| Sequential | 200 | 0.250 |

**分析**

與 Baseline 1 相比：

| Metric | Baseline 1 | Baseline 2 | 變化 |
|---|---|---|---|
| Precision | 0.011 | 0.398 | **+36x** |
| Recall | 0.873 | 0.341 | -60% |
| F1 | 0.022 | 0.367 | **+17x** |

Precision 大幅提升（36x）。Recall 下降的原因分析：

- **Gate 層 recall = 0.795**（245/308 GT pairs 通過所有 gate），表示 gate 設計合理。
- **Clustering 層 recall = 0.341**，遠低於 gate 層。
- 差距來源：compat set 中有 93.5% 為 FP pairs（3537/3782）。在 greedy matching 中，FP pairs 若 spatial score 較低（endpoint 距離更近），會搶先消耗 camera slot，導致某些 GT pairs 因 camera-conflict constraint 而無法合併。
- Sequential recall（0.250）明顯低於 Overlapping recall（0.509），因為 sequential pairs 只有 endpoint 距離作為 spatial 訊號，空間判別力弱於重疊對的 per-frame 距離。

這個分析直接指出 Baseline 3 的動機：在純幾何特徵無法充分區分 GT 與 FP pairs 的情況下，需要加入 appearance（Re-ID）特徵提供更強的判別訊號。

---

## Baseline 3 — Re-ID Enhanced（規劃中，尚未實作）

### 設計方向

在 Baseline 2 的 gate 框架上，加入 appearance affinity 作為額外的 association 訊號：

| 組件 | 說明 |
|---|---|
| Re-ID feature extractor | 從每個 tracklet 的 bbox crops 提取 appearance embedding（預計使用 OSNet 或類似輕量模型） |
| Affinity matrix | 整合 temporal / spatial / appearance cosine similarity |
| Matcher | 在 affinity 高於閾值的 pairs 上執行 association |

### 預期解決的問題

Baseline 2 仍無法區分「同時間、同位置、同 class 但不同個體」的 FP（如路口同向行駛的兩輛同款車）。Appearance feature 可在 spatial gate 無法分辨時提供額外判別力。

---

## 執行方式

```bash
# 安裝依賴
pip install pandas

# 執行 Baseline 1 / 2 並評估
python run_baseline.py

# 分析 GT 距離分布（calibrate thresholds）
python analyze_pair_dist.py
```
