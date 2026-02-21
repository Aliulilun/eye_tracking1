"""
第一階段：人臉檢測與特徵點定位
Stage 1: Face Detection and Landmark Localization

使用 MediaPipe Face Mesh 進行：
- 人臉檢測
- 提取 468 個面部特徵點
- 選擇關鍵點用於後續處理

作者: [你的名字]
日期: 2026-01
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, Optional, List, Tuple
import tempfile
import urllib.request
import os


class FaceDetector:
    """
    使用 MediaPipe Face Mesh 進行人臉檢測與特徵點定位
    """
    
    # MediaPipe 468 個特徵點中的關鍵點索引
    # 這些點對應於頭部姿態估計所需的位置
    KEY_LANDMARKS_INDICES = {
        'left_eye_outer': 33,      # 左眼外角
        'left_eye_inner': 133,     # 左眼內角
        'right_eye_inner': 362,    # 右眼內角
        'right_eye_outer': 263,    # 右眼外角
        'nose_tip': 1,             # 鼻尖
        'nose_bottom': 2,          # 鼻底（鼻孔之間）
        'left_mouth': 61,          # 左嘴角
        'right_mouth': 291,        # 右嘴角
    }
    
    def __init__(self, config: Dict = None):
        """
        初始化人臉檢測器
        
        Args:
            config: 配置字典，包含以下參數：
                - min_confidence: 最小檢測置信度 (0.0-1.0)
                - max_num_faces: 最大檢測人臉數量
                - model_selection: 模型選擇 (0: 近距離, 1: 全距離)
                - refine_landmarks: 是否精細化特徵點
        """
        if config is None:
            config = {}
        
        # 配置參數
        self.min_confidence = config.get('min_confidence', 0.5)
        self.max_num_faces = config.get('max_num_faces', 1)
        self.model_selection = config.get('model_selection', 1)
        self.refine_landmarks = config.get('refine_landmarks', True)
        
        # 獲取自定義關鍵點索引（如果配置中有）
        self.key_indices = config.get('key_landmarks_indices', None)
        if self.key_indices is None:
            # 使用默認的關鍵點順序（用於 solvePnP）
            self.key_indices = [
                self.KEY_LANDMARKS_INDICES['left_eye_outer'],
                self.KEY_LANDMARKS_INDICES['left_eye_inner'],
                self.KEY_LANDMARKS_INDICES['right_eye_inner'],
                self.KEY_LANDMARKS_INDICES['right_eye_outer'],
                self.KEY_LANDMARKS_INDICES['nose_tip'],
                self.KEY_LANDMARKS_INDICES['nose_bottom'],
                self.KEY_LANDMARKS_INDICES['left_mouth'],
                self.KEY_LANDMARKS_INDICES['right_mouth'],
            ]
        
        # 初始化 MediaPipe Face Landmarker (新版 API)
        # 下載模型文件（如果不存在）
        model_path = self._get_model_path()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_confidence,
            min_face_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        print(f"✓ FaceDetector 初始化完成 (MediaPipe {mp.__version__})")
        print(f"  - 最小置信度: {self.min_confidence}")
        print(f"  - 最大人臉數: {self.max_num_faces}")
    
    def _get_model_path(self) -> str:
        """下載並返回 Face Landmarker 模型路徑"""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            print("  - 下載 Face Landmarker 模型...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"  - 模型已下載到: {model_path}")
            except Exception as e:
                print(f"  - 警告：無法下載模型 ({e})")
                print("  - 請手動下載模型文件並放置到 models/face_landmarker.task")
                raise
        
        return model_path
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        檢測圖像中的人臉並提取特徵點
        
        Args:
            image: 輸入圖像（BGR 格式）
        
        Returns:
            如果檢測到人臉，返回字典包含：
                - 'bbox': 人臉邊界框 [x_min, y_min, x_max, y_max]
                - 'landmarks_468': 所有 478 個特徵點 (478, 2)
                - 'landmarks_2d_selected': 選定的關鍵點 (N, 2)
                - 'confidence': 檢測置信度
            如果未檢測到人臉，返回 None
        """
        # 轉換為 RGB（MediaPipe 需要 RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 創建 MediaPipe Image 對象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 獲取圖像尺寸
        h, w, _ = image.shape
        
        # 處理圖像
        detection_result = self.face_landmarker.detect(mp_image)
        
        # 檢查是否檢測到人臉
        if not detection_result.face_landmarks:
            return None
        
        # 取第一個檢測到的人臉（假設每張圖只有一個人臉）
        face_landmarks = detection_result.face_landmarks[0]
        
        # 提取所有特徵點的 2D 座標（新版有 478 個點）
        num_landmarks = len(face_landmarks)
        landmarks_all = np.zeros((num_landmarks, 2), dtype=np.float32)
        for idx, landmark in enumerate(face_landmarks):
            # 將歸一化座標轉換為像素座標
            landmarks_all[idx, 0] = landmark.x * w
            landmarks_all[idx, 1] = landmark.y * h
        
        # 對於向後兼容，我們仍然使用前 468 個點
        landmarks_468 = landmarks_all[:468] if num_landmarks >= 468 else landmarks_all
        
        # 計算人臉邊界框
        bbox = self._calculate_bbox(landmarks_468)
        
        # 選擇關鍵特徵點（用於頭部姿態估計）
        landmarks_2d_selected = landmarks_468[self.key_indices]
        
        # 計算置信度（使用 z 座標的平均值作為簡單的置信度指標）
        z_values = [landmark.z for landmark in face_landmarks]
        confidence = 1.0 - min(abs(np.mean(z_values)), 1.0)
        
        return {
            'bbox': bbox,
            'landmarks_468': landmarks_468,
            'landmarks_2d_selected': landmarks_2d_selected,
            'confidence': confidence,
            'num_landmarks': len(landmarks_468),
        }
    
    def _calculate_bbox(self, landmarks: np.ndarray) -> List[float]:
        """
        根據特徵點計算人臉邊界框
        
        Args:
            landmarks: 特徵點座標 (N, 2)
        
        Returns:
            邊界框 [x_min, y_min, x_max, y_max]
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = float(np.min(x_coords))
        y_min = float(np.min(y_coords))
        x_max = float(np.max(x_coords))
        y_max = float(np.max(y_coords))
        
        # 添加一些邊距（10%）
        width = x_max - x_min
        height = y_max - y_min
        margin = 0.1
        
        x_min = max(0, x_min - width * margin)
        y_min = max(0, y_min - height * margin)
        x_max = x_max + width * margin
        y_max = y_max + height * margin
        
        return [x_min, y_min, x_max, y_max]
    
    def get_landmark_by_name(self, landmarks_468: np.ndarray, 
                            landmark_name: str) -> np.ndarray:
        """
        根據名稱獲取特定特徵點
        
        Args:
            landmarks_468: 所有 468 個特徵點
            landmark_name: 特徵點名稱（如 'nose_tip', 'left_eye_outer' 等）
        
        Returns:
            特徵點座標 [x, y]
        """
        if landmark_name not in self.KEY_LANDMARKS_INDICES:
            raise ValueError(f"未知的特徵點名稱: {landmark_name}")
        
        idx = self.KEY_LANDMARKS_INDICES[landmark_name]
        return landmarks_468[idx]
    
    def visualize_landmarks(self, image: np.ndarray, 
                          landmarks_468: np.ndarray = None,
                          landmarks_selected: np.ndarray = None,
                          bbox: List[float] = None,
                          show_all: bool = False) -> np.ndarray:
        """
        在圖像上可視化特徵點和邊界框
        
        Args:
            image: 輸入圖像
            landmarks_468: 所有 468 個特徵點（可選）
            landmarks_selected: 選定的關鍵點（可選）
            bbox: 人臉邊界框（可選）
            show_all: 是否顯示所有 468 個點（默認只顯示選定的點）
        
        Returns:
            標註後的圖像
        """
        vis_image = image.copy()
        
        # 繪製邊界框
        if bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), 
                         (0, 255, 0), 2)
            cv2.putText(vis_image, 'Face', (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 繪製所有 468 個點（如果需要）
        if show_all and landmarks_468 is not None:
            for point in landmarks_468:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 1, (200, 200, 200), -1)
        
        # 繪製選定的關鍵點（更明顯）
        if landmarks_selected is not None:
            for idx, point in enumerate(landmarks_selected):
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 4, (0, 255, 0), -1)
                cv2.circle(vis_image, (x, y), 5, (0, 0, 255), 1)
                # 可選：添加點的索引標籤
                # cv2.putText(vis_image, str(idx), (x+5, y-5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return vis_image
    
    def get_eye_landmarks(self, landmarks_468: np.ndarray) -> Dict[str, np.ndarray]:
        """
        獲取眼睛相關的特徵點
        
        Args:
            landmarks_468: 所有 468 個特徵點
        
        Returns:
            字典包含左右眼的特徵點
        """
        # MediaPipe 眼睛輪廓的特徵點索引
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 
                           173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 
                            263, 466, 388, 387, 386, 385, 384, 398]
        
        return {
            'left_eye': landmarks_468[left_eye_indices],
            'right_eye': landmarks_468[right_eye_indices],
            'left_eye_center': np.mean(landmarks_468[left_eye_indices], axis=0),
            'right_eye_center': np.mean(landmarks_468[right_eye_indices], axis=0),
        }
    
    def get_face_orientation_estimate(self, landmarks_468: np.ndarray) -> Dict[str, float]:
        """
        基於特徵點的簡單人臉朝向估計（粗略估計）
        
        Args:
            landmarks_468: 所有 468 個特徵點
        
        Returns:
            字典包含估計的角度（度）
        """
        # 獲取關鍵點
        left_eye = landmarks_468[self.KEY_LANDMARKS_INDICES['left_eye_outer']]
        right_eye = landmarks_468[self.KEY_LANDMARKS_INDICES['right_eye_outer']]
        nose = landmarks_468[self.KEY_LANDMARKS_INDICES['nose_tip']]
        
        # 計算眼睛中點
        eye_center = (left_eye + right_eye) / 2
        
        # 估計 yaw（左右轉頭）
        # 基於鼻尖相對於眼睛中點的水平偏移
        eye_width = np.linalg.norm(right_eye - left_eye)
        nose_offset_x = nose[0] - eye_center[0]
        yaw_estimate = np.rad2deg(np.arctan2(nose_offset_x, eye_width))
        
        # 估計 pitch（上下點頭）
        # 基於鼻尖相對於眼睛的垂直偏移
        nose_offset_y = nose[1] - eye_center[1]
        pitch_estimate = np.rad2deg(np.arctan2(nose_offset_y, eye_width))
        
        return {
            'yaw_estimate': yaw_estimate,
            'pitch_estimate': pitch_estimate,
        }
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()


# ==================== 測試和示例代碼 ====================

def test_face_detector(image_path: str, output_path: str = None):
    """
    測試人臉檢測器
    
    Args:
        image_path: 輸入圖像路徑
        output_path: 輸出圖像路徑（可選）
    """
    print("=" * 70)
    print("測試第一階段：人臉檢測與特徵點定位")
    print("=" * 70)
    
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖像 {image_path}")
        return
    
    print(f"\n讀取圖像: {image_path}")
    print(f"圖像尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 創建檢測器
    config = {
        'min_confidence': 0.5,
        'max_num_faces': 1,
        'refine_landmarks': True,
    }
    detector = FaceDetector(config=config)
    
    # 檢測人臉
    print("\n執行人臉檢測...")
    result = detector.detect(image)
    
    if result is None:
        print("❌ 未檢測到人臉！")
        return
    
    print("✓ 檢測到人臉！")
    print(f"\n檢測結果:")
    print(f"  - 總特徵點數: {result['num_landmarks']}")
    print(f"  - 選定關鍵點數: {len(result['landmarks_2d_selected'])}")
    print(f"  - 檢測置信度: {result['confidence']:.3f}")
    print(f"  - 人臉邊界框: [{result['bbox'][0]:.1f}, {result['bbox'][1]:.1f}, "
          f"{result['bbox'][2]:.1f}, {result['bbox'][3]:.1f}]")
    
    # 顯示關鍵特徵點座標
    print(f"\n關鍵特徵點座標:")
    landmark_names = ['左眼外角', '左眼內角', '右眼內角', '右眼外角', 
                     '鼻尖', '鼻底', '左嘴角', '右嘴角']
    for idx, (name, point) in enumerate(zip(landmark_names, 
                                            result['landmarks_2d_selected'])):
        print(f"  {idx+1}. {name:8s}: ({point[0]:7.2f}, {point[1]:7.2f})")
    
    # 簡單的人臉朝向估計
    orientation = detector.get_face_orientation_estimate(result['landmarks_468'])
    print(f"\n簡單人臉朝向估計:")
    print(f"  - Yaw (左右): {orientation['yaw_estimate']:>7.2f}°")
    print(f"  - Pitch (上下): {orientation['pitch_estimate']:>7.2f}°")
    
    # 可視化
    vis_image = detector.visualize_landmarks(
        image,
        landmarks_468=result['landmarks_468'],
        landmarks_selected=result['landmarks_2d_selected'],
        bbox=result['bbox'],
        show_all=False  # 設為 True 可顯示所有 468 個點
    )
    
    # 保存或顯示結果
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"\n✓ 結果已保存到: {output_path}")
    
    # 顯示結果
    cv2.imshow('Face Detection Result', vis_image)
    print("\n按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 從命令行參數讀取圖像路徑
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'output/stage1_result.jpg'
        test_face_detector(image_path, output_path)
    else:
        print("用法: python stage1_face_detection.py <image_path> [output_path]")
        print("\n範例:")
        print("  python stages/stage1_face_detection.py test_images/face.jpg")
        print("  python stages/stage1_face_detection.py test_images/face.jpg output/result.jpg")

