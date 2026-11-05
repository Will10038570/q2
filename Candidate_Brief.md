# 二面 Side Project：多攝影機道路使用者軌跡關聯

## 1. 背景

某一路口設置多支 CCTV。每支攝影機目前均可獨立完成：

- Object Detection
- Single-Camera Tracking

因此，每支攝影機會產生一組 local tracklets。

同一個道路使用者可能同時或先後出現在不同攝影機中，因此不同攝影機所產生的 local tracklets 可能實際對應到同一個實體。

本題希望你設計一個方法，將不同攝影機中的 local tracklets 進行關聯，建立一致的跨攝影機 global identity。

---

## 2. 任務

請設計並實作一個 **Multi-Camera Tracklet Association** 方法。

你的系統需判斷：

> 不同攝影機產生的兩段或多段 local tracklets，是否代表同一個道路使用者？

若判斷為同一實體，請將它們指派為相同的 global identity。

### Input

主要資料位於：

```text
dev/tracklets.csv
test/tracklets.csv
```

欄位包含：

- `camera_id`
- `frame_id`
- `timestamp`
- `local_track_id`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `class`
- `confidence`
- `world_x`
- `world_y`
- `world_valid`

其中：

- `timestamp` 已轉換為跨攝影機共用的時間基準，但仍可能存在小幅同步誤差。
- `world_x / world_y` 為共同的局部世界座標，單位為公尺。
- `world_valid = 1` 表示該 observation 的世界座標可使用。
- `world_valid = 0` 時，`world_x / world_y` 為缺失值，不應視為可靠位置資訊。

資料中可能包含：

- track fragmentation
- missed observations
- duplicate / distractor tracks
- localization error
- 小幅 timestamp synchronization error
- 相似外觀的不同道路使用者
- 部分攝影機視野重疊、部分不重疊

請勿假設每個 local track 都一定能在其他攝影機中找到對應 track。

---

## 3. Development / Test

### Development Set

```text
dev/tracklets.csv
dev/ground_truth.csv
```

`ground_truth.csv` 提供部分可靠 local tracks 的 global identity，可用於：

- 方法開發
- parameter tuning
- evaluation
- error analysis

未列在 `ground_truth.csv` 的 local tracks 應視為 unlabeled / distractor tracks，不應直接假設為某個 ground-truth identity。

### Hidden Test Set

```text
test/tracklets.csv
```

Test set 不提供 global identity。

請對 test set 輸出：

```text
camera_id,local_track_id,predicted_global_id
```

我們將使用 hidden ground truth 評估結果。

---

## 4. 方法限制

**不限制方法。**

你可以使用：

- Rule-based methods
- Optimization
- Traditional Computer Vision
- Machine Learning
- Deep Learning
- 或上述方法的組合

不要求使用特定 framework、模型或演算法。

我們更重視：

- 你如何定義問題
- 你如何建立 baseline
- 你如何選擇下一個改進方向
- 你如何設計實驗來驗證假設
- 你如何分析 failure cases
- 你如何評估工程上的 trade-off

而不是單純追求模型複雜度。

---

## 5. 評估

Package 中提供：

```text
eval.py
```

可用於 development set 的基本 association evaluation。

你可以使用該 evaluator，也可以補充其他你認為更合理的 metrics。

請在報告中說明：

1. 你選擇哪些 evaluation metrics
2. 為什麼選擇這些 metrics
3. metrics 分別衡量系統的哪一種能力
4. 哪些 failure cases 不容易由單一 aggregate metric 反映

至少需要評估：

- cross-camera identity association correctness
- 方法在不同 failure cases 下的表現

若你認為 trajectory quality、latency、memory usage 或其他指標也很重要，可自行補充。

---

## 6. 實驗設計

請自行規劃合理的實驗。

我們不指定實驗數量，也不要求固定流程。

希望從你的結果中可以看出：

- 最初 baseline 是什麼
- baseline 的限制是什麼
- 你如何根據 error analysis 決定下一步
- 新增的 component 是否對應到明確假設
- 各方法之間的 trade-off
- 最終方法仍有哪些 failure modes

若使用多個資訊來源或多個 components，建議能說明各 component 對結果的實際影響。

---

## 7. Engineering Consideration

最終系統預計用於即時 CCTV stream。

本題不要求完成 production optimization，但請分析：

- latency
- throughput
- memory usage
- camera 數量增加時的 scalability
- missing / delayed frames
- timestamp synchronization error
- world-coordinate uncertainty
- upstream detector / tracker error

請指出目前 prototype 中最可能成為瓶頸的部分，以及如果要進一步產品化，你會如何驗證與改善。

---

## 8. Deliverables

### A. Technical Report / README

至少包含：

- Problem definition
- Assumptions
- Method
- Baseline
- Experiment design
- Evaluation metrics
- Results
- Error analysis
- Limitations
- 下一步改善方向

形式可為 Markdown、PDF 或其他易讀格式。

### B. Source Code

要求：

- 可執行
- 結構清楚
- 能重現主要 development 結果
- 提供必要的環境與執行說明

不要求完整 UI。

### C. Test Prediction

請提供：

```text
test_predictions.csv
```

格式：

```text
camera_id,local_track_id,predicted_global_id
```

### D. 二面簡報

請準備約 **10–15 分鐘**簡報，說明：

1. 你怎麼理解這個問題
2. 你的 baseline
3. 方法與設計理由
4. 實驗與主要結果
5. 重要 failure cases
6. 如果再多一週，你下一步會做什麼

---

## 9. 時間與資源

建議投入時間：約 **30 小時以內**。

本題不期待：

- 完整 production system
- 大規模模型訓練
- SOTA performance
- 完整 UI

請優先確保：

> 問題定義清楚 → baseline 可重現 → 實驗能驗證假設 → 結果能被合理解釋

若使用 pretrained model 或外部公開資源，請在報告中註明來源。若需要 GPU 資源，可考慮 Google Colab。

---

## 10. 補充說明

這是一個 open-ended engineering / ML problem。

沒有唯一正確解法。

即使最終 test performance 不是最高，只要你能清楚說明：

- 為什麼選擇目前的方法
- 哪些假設成立、哪些不成立
- 哪些結果支持或反駁原先假設
- 哪些 failure cases 尚未解決

都會是評估的重要部分。
