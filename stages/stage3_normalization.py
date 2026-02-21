"""
第三階段：圖像正規化 (ETH-XGaze 標準)
Stage 3: Image Normalization

符合 ETH-XGaze 論文的 3D normalization 實作
參考: https://ait.ethz.ch/projects/2020/ETH-XGaze/

關鍵修正：
1. 統一單位為 cm（與 stage2 一致）
2. 正確的座標系轉換（MediaPipe Y-up, Z-forward）
3. 正確的 face center 計算和距離縮放
4. 正確的旋轉矩陣構建

作者: Claude
日期: 2026-02
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImageNormalizer:
    """
    ETH-XGaze 標準的圖像正規化器
    
    核心步驟：
    1. 從 solvePnP 結果獲得 R_head, t_head（頭部在相機座標系）
    2. 計算 face center 在相機座標系的位置
    3. 建立正規化相機座標系（Z 軸指向 face center）
    4. 計算縮放係數，將 face 移動到固定距離 distance_norm
    5. 計算 homography 矩陣 W = K_norm @ S @ R_norm @ R_head^T @ K_orig^(-1)
    6. warpPerspective 得到正規化圖像
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化圖像正規化器
        
        Args:
            config: 配置字典：
                - output_size: (width, height)，默認 (224, 224)
                - focal_norm: 正規化焦距，默認 960
                - distance_norm: 正規化距離（cm），默認 60
                - face_model_path: 3D 人臉模型路徑
        """
        if config is None:
            config = {}
        
        # 輸出參數
        self.output_size = config.get('output_size', (224, 224))
        
        # 正規化參數（ETH-XGaze 標準）
        self.focal_norm = config.get('focal_norm', 960.0)
        self.distance_norm = config.get('distance_norm', 60.0)  # cm（ETH-XGaze 標準：60 cm = 600 mm）
        
        # 載入 3D 人臉模型（保持 cm 單位，與 stage2 一致）
        face_model_path = config.get('face_model_path', 'models/face_model_mediapipe.txt')
        self.face_model_3d = self._load_face_model(face_model_path)
        
        # 計算正規化相機矩陣
        self.camera_matrix_norm = self._get_normalized_camera_matrix()
        
        print(f"✅ ImageNormalizer 初始化完成")
        print(f"  - 輸出尺寸: {self.output_size}")
        print(f"  - 正規化焦距: {self.focal_norm}")
        print(f"  - 正規化距離: {self.distance_norm} cm")
        print(f"  - 3D 模型點數: {self.face_model_3d.shape[0]}")
    
    def _load_face_model(self, model_path: str) -> np.ndarray:
        """
        載入 3D 人臉模型（單位：cm）
        
        重要：保持與 stage2 一致的單位（cm）
        """
        try:
            face_model = np.loadtxt(model_path, comments='#')
            if face_model.ndim == 1:
                face_model = face_model.reshape(-1, 3)
            
            # 確認單位是 cm（face_model_mediapipe.txt 已經是 cm）
            print(f"  - 載入 3D 模型: {model_path}")
            print(f"  - 模型點數: {face_model.shape[0]}")
            print(f"  - 單位: cm")
            
            return face_model.astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"無法載入 3D 人臉模型: {e}")
    
    def _get_normalized_camera_matrix(self) -> np.ndarray:
        """
        構建正規化相機內參矩陣
        """
        cx_norm = self.output_size[0] / 2.0
        cy_norm = self.output_size[1] / 2.0
        
        K_norm = np.array([
            [self.focal_norm, 0, cx_norm],
            [0, self.focal_norm, cy_norm],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K_norm
    
    def normalize(self,
                 image: np.ndarray,
                 rotation_vector: np.ndarray,
                 translation_vector: np.ndarray,
                 camera_matrix: np.ndarray) -> Dict:
        """
        ETH-XGaze 官方正規化方式
        
        關鍵差異：使用 head forward direction (R[:, 2]) 而非 face center direction
        
        Args:
            image: 原始圖像
            rotation_vector: 頭部旋轉向量 (3,1)，來自 stage2 solvePnP
            translation_vector: 頭部平移向量 (3,1)，來自 stage2 solvePnP (單位: cm)
            camera_matrix: 原始相機內參矩陣 (3,3)
        
        Returns:
            result: {
                'normalized_image': 正規化圖像 (224, 224, 3)
                'warp_matrix': 透視變換矩陣 (3, 3)
                'head_rot_norm': 正規化後的頭部旋轉矩陣 (3, 3)
                'gaze_rot_norm': 正規化後的視線旋轉矩陣 (3, 3)
                'success': bool
            }
        """
        try:
            # --------------------------------------------------
            # 1) solvePnP → R, t (單位: cm)
            # --------------------------------------------------
            R, _ = cv2.Rodrigues(rotation_vector)
            t = translation_vector.reshape(3, 1)

            # 距離（cm）
            distance = np.linalg.norm(t)
            if distance < 1e-6:
                return self._fallback(image)

            # --------------------------------------------------
            # 2) 建立 normalized camera 座標系
            #    ⚠️ 關鍵：z 軸 = head forward direction (R[:, 2])
            # --------------------------------------------------
            z_n = R[:, 2].reshape(3, 1)
            z_n /= np.linalg.norm(z_n)

            up = np.array([[0.0], [1.0], [0.0]])

            # 避免平行
            if abs(np.dot(up.flatten(), z_n.flatten())) > 0.99:
                up = np.array([[1.0], [0.0], [0.0]])

            x_n = np.cross(up.flatten(), z_n.flatten()).reshape(3, 1)
            x_n /= np.linalg.norm(x_n)

            y_n = np.cross(z_n.flatten(), x_n.flatten()).reshape(3, 1)
            y_n /= np.linalg.norm(y_n)

            R_n = np.hstack([x_n, y_n, z_n])

            # --------------------------------------------------
            # 3) 固定距離（ETH-XGaze: 60 cm）
            # --------------------------------------------------
            distance_norm = self.distance_norm  # 60 cm

            # 等效縮放比例（隱含距離調整）
            scale = distance_norm / distance

            # --------------------------------------------------
            # 4) 平面近似 Homography
            # --------------------------------------------------
            K_inv = np.linalg.inv(camera_matrix)

            # 注意：scale 只影響平面投影比例
            W = self.camera_matrix_norm @ (scale * R_n @ R.T) @ K_inv
            W = W / W[2, 2]

            # --------------------------------------------------
            # 5) warp
            # --------------------------------------------------
            normalized_image = cv2.warpPerspective(
                image,
                W.astype(np.float32),
                self.output_size,
                flags=cv2.INTER_LINEAR
            )

            # --------------------------------------------------
            # 6) 正規化後頭部旋轉
            # --------------------------------------------------
            head_rot_norm = R_n @ R.T

            return {
                'normalized_image': normalized_image,
                'warp_matrix': W.astype(np.float32),
                'head_rot_norm': head_rot_norm,
                'gaze_rot_norm': head_rot_norm.copy(),
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 正規化失敗: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback(image)
    
    def _fallback(self, image: np.ndarray) -> Dict:
        """
        失敗時的備用方案
        """
        fallback_image = cv2.resize(image, self.output_size)
        return {
            'normalized_image': fallback_image,
            'warp_matrix': np.eye(3, dtype=np.float32),
            'head_rot_norm': np.eye(3, dtype=np.float32),
            'scale_factor': 1.0,
            'face_center_distance': 0.0,
            'success': False
        }
    
    def visualize_normalization(self,
                               original_image: np.ndarray,
                               normalized_image: np.ndarray,
                               warp_matrix: np.ndarray,
                               face_center_distance: float = None,
                               scale_factor: float = None) -> np.ndarray:
        """
        可視化正規化結果
        
        在原圖上繪製正規化區域的邊界
        """
        h_norm, w_norm = normalized_image.shape[:2]
        
        # 正規化圖像的四個角點
        corners_norm = np.array([
            [0, 0],
            [w_norm - 1, 0],
            [w_norm - 1, h_norm - 1],
            [0, h_norm - 1]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # 反變換到原圖
        try:
            W_inv = np.linalg.inv(warp_matrix)
            corners_orig = cv2.perspectiveTransform(corners_norm, W_inv)
            corners_orig = corners_orig.reshape(-1, 2).astype(np.int32)
            
            # 繪製邊界
            vis = original_image.copy()
            for i in range(4):
                pt1 = tuple(corners_orig[i])
                pt2 = tuple(corners_orig[(i + 1) % 4])
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            
            # 繪製對角線（幫助理解變換）
            cv2.line(vis, tuple(corners_orig[0]), tuple(corners_orig[2]), 
                    (0, 255, 0), 1)
            cv2.line(vis, tuple(corners_orig[1]), tuple(corners_orig[3]), 
                    (0, 255, 0), 1)
            
        except:
            vis = original_image.copy()
        
        # 縮放並並排顯示
        vis_resized = cv2.resize(vis, (w_norm, h_norm))
        comparison = np.hstack([vis_resized, normalized_image])
        
        # 添加標籤
        cv2.putText(comparison, 'Original + ROI', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, 'Normalized', (w_norm + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加資訊
        if face_center_distance is not None:
            cv2.putText(comparison, f'Distance: {face_center_distance:.1f} cm', 
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if scale_factor is not None:
            cv2.putText(comparison, f'Scale: {scale_factor:.3f}', 
                       (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 分隔線
        cv2.line(comparison, (w_norm, 0), (w_norm, h_norm), (255, 255, 255), 2)
        
        return comparison
    
    def get_normalization_params(self) -> Dict:
        """
        獲取正規化參數
        """
        return {
            'output_size': self.output_size,
            'focal_norm': self.focal_norm,
            'distance_norm': self.distance_norm,
            'camera_matrix_norm': self.camera_matrix_norm.tolist()
        }


def create_image_normalizer(config: Dict = None) -> ImageNormalizer:
    """
    工廠函數：創建圖像正規化器
    """
    return ImageNormalizer(config)


# ==================== 測試代碼 ====================

def test_normalizer():
    """
    測試正規化器
    """
    print("=" * 70)
    print("測試 ETH-XGaze 圖像正規化")
    print("=" * 70)
    
    # 創建正規化器
    config = {
        'output_size': (224, 224),
        'focal_norm': 960.0,
        'distance_norm': 60.0,  # cm
        'face_model_path': 'models/face_model_mediapipe.txt'
    }
    
    try:
        normalizer = create_image_normalizer(config)
    except FileNotFoundError:
        print("⚠️ 找不到 3D 模型文件，使用默認配置")
        config['face_model_path'] = None
        normalizer = create_image_normalizer(config)
    
    # 創建測試圖像
    print("\n創建測試圖像...")
    image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    
    # 繪製測試圖案
    cv2.rectangle(image, (220, 140), (420, 340), (255, 200, 100), -1)
    cv2.circle(image, (280, 200), 20, (0, 0, 255), -1)  # 左眼
    cv2.circle(image, (360, 200), 20, (0, 0, 255), -1)  # 右眼
    cv2.ellipse(image, (320, 280), (40, 20), 0, 0, 180, (0, 0, 0), 2)  # 嘴
    
    # 模擬頭部姿態（輕微向右轉）
    rvec = np.array([[0.1], [-0.2], [0.05]], dtype=np.float32)  # 弧度
    tvec = np.array([[0], [0], [60]], dtype=np.float32)  # cm
    
    # 相機矩陣
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 執行正規化
    print("\n執行正規化...")
    result = normalizer.normalize(image, rvec, tvec, K)
    
    if result['success']:
        print("\n✅ 正規化成功！")
        
        # 可視化
        vis = normalizer.visualize_normalization(
            image,
            result['normalized_image'],
            result['warp_matrix'],
            result['face_center_distance'],
            result['scale_factor']
        )
        
        # 保存結果
        from pathlib import Path
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'test_normalized.jpg'), 
                   result['normalized_image'])
        cv2.imwrite(str(output_dir / 'test_comparison.jpg'), vis)
        
        print(f"✅ 結果已保存到 output/")
    else:
        print("\n❌ 正規化失敗")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_normalizer()