"""
第二階段：頭部姿態估計
Stage 2: Head Pose Estimation

使用 OpenCV solvePnP + scipy Rotation
簡化版實作，直接從旋轉矩陣計算 Euler 角

作者: [你的名字]
日期: 2026-02
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class HeadPoseEstimator:
    """
    使用 OpenCV solvePnP 進行頭部姿態估計
    
    核心邏輯：
    - MediaPipe 3D 模型 + 2D landmarks → solvePnP → rvec/tvec
    - rvec → rotation matrix → Euler angles (YXZ 順序)
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化頭部姿態估計器
        
        Args:
            config: 配置字典，包含以下鍵值：
                - face_model_path: 3D 人臉模型路徑
        """
        if config is None:
            config = {}
        
        # 3D 人臉模型路徑
        self.face_model_path = config.get(
            'face_model_path',
            'models/face_model_mediapipe.txt'
        )
        
        # 載入 3D 人臉模型
        self.face_model_3d = self._load_face_model()
        
        print(f"✅ 頭部姿態估計器初始化完成")
        print(f"  - 3D 模型點數: {self.face_model_3d.shape[0]}")
    
    def _load_face_model(self) -> np.ndarray:
        """
        載入 3D 人臉模型（MediaPipe canonical face model）
        """
        try:
            face_model = np.loadtxt(self.face_model_path, comments='#')
            if face_model.ndim == 1:
                face_model = face_model.reshape(-1, 3)
            
            # 轉換為 float32（OpenCV 要求）
            face_model = face_model.astype(np.float32)
            
            print(f"  - 載入 3D 人臉模型: {self.face_model_path}")
            print(f"  - 模型點數: {face_model.shape[0]}")
            
            return face_model
        except Exception as e:
            raise FileNotFoundError(f"無法載入 3D 人臉模型: {e}")
    
    def estimate(self, 
                 landmarks_2d: np.ndarray, 
                 camera_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray = None) -> Dict:
        """
        估計頭部姿態
        
        Args:
            landmarks_2d: 2D 特徵點 (N, 2)，像素座標
            camera_matrix: 相機內參矩陣 (3, 3)
            distortion_coeffs: 畸變係數（可選）
        
        Returns:
            result: 包含姿態信息的字典：
                - 'rvec': 旋轉向量 (3, 1)
                - 'tvec': 平移向量 (3, 1)
                - 'rotation_matrix': 旋轉矩陣 (3, 3)
                - 'euler_angles': 歐拉角字典 {'pitch': float, 'yaw': float, 'roll': float}（度）
                - 'success': 是否成功求解
        """
        # 檢查輸入
        if landmarks_2d.shape[0] != self.face_model_3d.shape[0]:
            raise ValueError(
                f"2D 特徵點數量 ({landmarks_2d.shape[0]}) "
                f"與 3D 模型點數量 ({self.face_model_3d.shape[0]}) 不匹配"
            )
        
        # 準備輸入數據
        if distortion_coeffs is None:
            distortion_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        object_points = self.face_model_3d.reshape(-1, 1, 3)
        image_points = landmarks_2d.reshape(-1, 1, 2).astype(np.float32)
        
        # 使用 solvePnP 求解姿態
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("⚠️ solvePnP 求解失敗")
            return {
                'rvec': np.zeros((3, 1)),
                'tvec': np.zeros((3, 1)),
                'rotation_matrix': np.eye(3),
                'euler_angles': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
                'success': False
            }
        
        # rvec → rotation matrix
        R_cam, _ = cv2.Rodrigues(rvec)
        
        # Rotation matrix → Euler angles（YXZ 順序）
        r = R.from_matrix(R_cam)
        yaw, pitch, roll = r.as_euler('YXZ', degrees=True)
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': R_cam,
            'euler_angles': {
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll)
            },
            'success': True
        }
    
    def draw_axes(self, 
                  image: np.ndarray, 
                  rvec: np.ndarray, 
                  tvec: np.ndarray,
                  camera_matrix: np.ndarray,
                  distortion_coeffs: np.ndarray = None,
                  axis_length: float = 50.0) -> np.ndarray:
        """
        在圖像上繪製 3D 坐標軸
        
        Args:
            image: 輸入圖像
            rvec: 旋轉向量
            tvec: 平移向量
            camera_matrix: 相機內參矩陣
            distortion_coeffs: 畸變係數
            axis_length: 坐標軸長度（像素）
        
        Returns:
            繪製了坐標軸的圖像
        """
        if distortion_coeffs is None:
            distortion_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 定義 3D 坐標軸端點
        axis_points_3d = np.array([
            [0, 0, 0],              # 原點
            [axis_length, 0, 0],    # X 軸（紅色）
            [0, axis_length, 0],    # Y 軸（綠色）
            [0, 0, axis_length]     # Z 軸（藍色）
        ], dtype=np.float32)
        
        # 投影到 2D
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d,
            rvec,
            tvec,
            camera_matrix,
            distortion_coeffs
        )
        
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        
        # 繪製坐標軸
        origin = tuple(axis_points_2d[0])
        image = cv2.line(image, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3)  # X - 紅色
        image = cv2.line(image, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3)  # Y - 綠色
        image = cv2.line(image, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3)  # Z - 藍色
        
        return image
    
    def get_head_direction_vector(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        從旋轉矩陣獲取頭部方向向量（視線方向）
        
        Args:
            rotation_matrix: 旋轉矩陣 (3, 3)
        
        Returns:
            direction_vector: 3D 單位向量 (3,)
        """
        # Z 軸方向即為頭部朝向
        direction_vector = rotation_matrix[:, 2]
        return direction_vector / np.linalg.norm(direction_vector)


def create_head_pose_estimator(config: Dict = None) -> HeadPoseEstimator:
    """
    工廠函數：創建頭部姿態估計器實例
    """
    return HeadPoseEstimator(config)


# 使用範例
if __name__ == "__main__":
    print("Stage 2: 頭部姿態估計模組")
    print("=" * 50)
    
    # 創建估計器
    estimator = create_head_pose_estimator()
    
    print("\n模組測試完成 ✅")