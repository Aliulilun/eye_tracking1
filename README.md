# 視線估計系統 (Gaze Estimation System)

基於 ETH-XGaze 預訓練模型的完整視線估計流程實現。

## 專案結構

```
eye_tracking/
├── README.md                           # 專案說明文件
├── config.yaml                         # 系統配置文件
├── requirements.txt                    # Python 依賴套件
│
├── my_gaze_estimation.py              # 🚀 主程式（整合所有階段）
│
├── stages/                            # 五個階段的實現
│   ├── __init__.py
│   ├── stage1_face_detection.py      # 第一階段：人臉檢測與特徵點定位
│   ├── stage2_head_pose.py           # 第二階段：頭部姿態估計
│   ├── stage3_normalization.py       # 第三階段：圖像正規化
│   ├── stage4_gaze_network.py        # 第四階段：神經網絡推理
│   └── stage5_gaze_vector.py         # 第五階段：視線向量轉換
│
├── utils/                             # 工具函數
│   ├── __init__.py
│   ├── visualization.py              # 可視化工具
│   └── camera_utils.py               # 相機參數處理
│
├── models/                            # 模型文件
│   ├── epoch_24_ckpt.pth.tar         # ETH-XGaze 預訓練模型
│   └── face_model_mediapipe.txt      # 3D 人臉模型（MediaPipe 對應）
│
├── reference/                         # 參考代碼
│   ├── demo.py                       # ETH-XGaze 原始 demo
│   └── normalization_example.py      # 原始正規化範例
│
├── test_images/                       # 測試圖片
│   └── (放置你的測試圖片)
│
└── output/                            # 輸出結果
    └── results/
        └── (生成的結果圖片)
```

## 五個階段說明

### 第一階段：人臉檢測與特徵點定位
- **工具**: MediaPipe Face Mesh
- **功能**: 偵測人臉框 + 提取 468 個面部特徵點
- **輸出**: 眼角、鼻尖、嘴角等關鍵點的 2D 座標

### 第二階段：頭部姿態估計
- **工具**: OpenCV 的 `solvePnP` 函數
- **功能**: 用 3D 人臉模型 + 2D 特徵點計算頭部旋轉矩陣和平移向量
- **輸出**: 6DoF 頭部姿態（3 個旋轉角 + 3 個平移量）

### 第三階段：圖像正規化
- **工具**: OpenCV 的透視變換 `warpPerspective`
- **功能**: 將人臉圖像轉換到「虛擬正面視角」，消除頭部旋轉影響
- **輸出**: 224×224 的正規化 RGB 人臉圖像

### 第四階段：神經網絡推理
- **工具**: ETH-XGaze 預訓練的 ResNet-50 模型
- **功能**: 從正規化圖像預測視線方向
- **輸出**: 視線的 pitch 和 yaw 角度（弧度）

### 第五階段：視線向量轉換
- **工具**: 簡單的三角函數計算
- **功能**: 將 (pitch, yaw) 轉成 3D 單位向量
- **公式**:
  - x = -cos(pitch) × sin(yaw)
  - y = -sin(pitch)
  - z = -cos(pitch) × cos(yaw)

## 安裝

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 下載預訓練模型

ETH-XGaze 預訓練模型 `epoch_24_ckpt.pth.tar` 應該已經放在 `models/` 目錄中。

如果沒有，請從 [ETH-XGaze GitHub](https://github.com/xucong-zhang/ETH-XGaze) 下載。

### 3. 準備 3D 人臉模型

系統會自動生成適用於 MediaPipe 的 3D 人臉模型文件。

## 使用方法

### 🖼️ 處理單張圖片

#### 基本使用

```bash
python my_gaze_estimation.py --image test_images/your_image.jpg
```

#### 指定輸出路徑

```bash
python my_gaze_estimation.py \
    --image test_images/your_image.jpg \
    --output output/results/result.jpg
```

#### 使用相機校正文件

```bash
python my_gaze_estimation.py \
    --image test_images/your_image.jpg \
    --camera-file camera_calibration.xml
```

#### 自定義配置

```bash
python my_gaze_estimation.py \
    --image test_images/your_image.jpg \
    --config my_config.yaml
```

### 🎬 處理影片（推薦用於研究）

#### 基本使用 - 處理影片並輸出結果

```bash
python process_video.py --input your_video.mp4 --output output.mp4 --csv gaze_data.csv
```

#### 僅導出數據（不輸出影片，速度更快）

```bash
python process_video.py --input your_video.mp4 --csv gaze_data.csv
```

#### 顯示即時預覽

```bash
python process_video.py --input your_video.mp4 --output output.mp4 --show-preview
```

#### 跳幀處理（提高處理速度）

```bash
# 每 3 幀處理 1 幀（速度提升 3 倍）
python process_video.py --input your_video.mp4 --skip-frames 2 --csv gaze_data.csv
```

#### 測試模式（只處理前 100 幀）

```bash
python process_video.py --input your_video.mp4 --max-frames 100 --csv test.csv
```

#### 繪製面部特徵點

```bash
python process_video.py --input your_video.mp4 --output output.mp4 --draw-landmarks
```

### 📊 輸出的 CSV 數據格式

處理影片後，CSV 文件會包含以下欄位：

- `frame_idx`: 幀索引
- `timestamp_sec`: 時間戳（秒）
- `head_pitch_deg`, `head_yaw_deg`, `head_roll_deg`: 頭部姿態角度
- `gaze_pitch_deg`, `gaze_yaw_deg`: 視線角度（度）
- `gaze_pitch_rad`, `gaze_yaw_rad`: 視線角度（弧度）
- `gaze_vector_x`, `gaze_vector_y`, `gaze_vector_z`: 3D 視線向量
- `face_bbox_x`, `face_bbox_y`, `face_bbox_w`, `face_bbox_h`: 人臉位置

## 配置說明

編輯 `config.yaml` 來自定義系統行為：

- **face_detection**: 調整 MediaPipe 檢測參數
- **normalization**: 調整圖像正規化參數
- **model**: 設置使用 GPU 或 CPU
- **output**: 控制結果保存和可視化

## 測試參考實現

如果想測試 ETH-XGaze 官方的實現：

```bash
cd reference
python demo.py
```

（需要額外下載 dlib 模型和範例圖片）

## 系統需求

- **Python**: 3.8+
- **作業系統**: macOS, Linux, Windows
- **GPU**: 可選（使用 CPU 也可以運行，但速度較慢）

## 常見問題

### Q: 未檢測到人臉？
- 確保圖像中有清晰可見的人臉
- 調整 `config.yaml` 中的 `min_confidence` 參數

### Q: 使用 GPU 時出錯？
- 確認已安裝對應版本的 PyTorch (CUDA)
- 或在 `config.yaml` 中將 `device` 改為 `"cpu"`

### Q: 視線估計結果不準確？
- 確保輸入圖像質量良好
- 檢查人臉是否被遮擋
- 使用相機校正文件可以提高精度

## 參考資料

- [ETH-XGaze Dataset](https://github.com/xucong-zhang/ETH-XGaze)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## 授權

本專案基於 ETH-XGaze 的預訓練模型，遵循其原始授權條款。

## 作者

- 專案實現：[你的名字]
- 基於：ETH-XGaze (Xucong Zhang et al., ECCV 2020)

