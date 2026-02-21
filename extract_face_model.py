"""
從 MediaPipe canonical face model 提取關鍵點座標
Extract key landmarks from MediaPipe canonical face model
"""

import numpy as np

def load_obj_vertices(obj_file):
    """載入 OBJ 文件中的所有頂點座標"""
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # v X Y Z
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)


def extract_mediapipe_keypoints():
    """
    提取 MediaPipe 的 8 個關鍵點
    
    MediaPipe 的 canonical face model 有 468 個頂點
    頂點索引對應 MediaPipe Face Mesh 的特徵點索引
    """
    # 載入官方模型
    vertices = load_obj_vertices('models/canonical_face_model.obj')
    
    print(f"載入了 {len(vertices)} 個頂點")
    print(f"頂點範圍:")
    print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
    print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
    print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
    
    # MediaPipe 關鍵點索引（對應 468 個特徵點）
    keypoint_indices = {
        'left_eye_outer': 33,      # 左眼外角
        'left_eye_inner': 133,     # 左眼內角
        'right_eye_inner': 362,    # 右眼內角
        'right_eye_outer': 263,    # 右眼外角
        'nose_tip': 1,             # 鼻尖
        'nose_bottom': 2,          # 鼻底
        'left_mouth': 61,          # 左嘴角
        'right_mouth': 291,        # 右嘴角
    }
    
    # 提取座標
    print("\n提取的關鍵點座標:")
    print("=" * 70)
    
    keypoints_3d = []
    for name, idx in keypoint_indices.items():
        if idx < len(vertices):
            coord = vertices[idx]
            keypoints_3d.append(coord)
            print(f"{name:20s} (index {idx:3d}): [{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}]")
        else:
            print(f"警告: 索引 {idx} 超出範圍")
    
    return np.array(keypoints_3d), keypoint_indices


def save_face_model(output_file='models/face_model_mediapipe.txt'):
    """保存為文本格式"""
    keypoints_3d, indices = extract_mediapipe_keypoints()
    
    with open(output_file, 'w') as f:
        f.write("# 3D 人臉模型 - MediaPipe 官方 Canonical Face Model\n")
        f.write("# 來源: https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/data\n")
        f.write("# Face Model 3D Coordinates for MediaPipe landmarks\n")
        f.write("# 格式: X Y Z (單位: cm，已從官方模型縮放)\n")
        f.write("# 座標系統: X 向右，Y 向上，Z 向前（遠離臉部）\n")
        f.write("#\n")
        f.write("# 注意: MediaPipe canonical model 的座標系:\n")
        f.write("#   - Y 軸向上為正\n")
        f.write("#   - Z 軸向前（遠離臉部）為正\n")
        f.write("#\n\n")
        
        names = [
            'left_eye_outer',
            'left_eye_inner', 
            'right_eye_inner',
            'right_eye_outer',
            'nose_tip',
            'nose_bottom',
            'left_mouth',
            'right_mouth'
        ]
        
        descriptions = [
            'Left eye outer corner',
            'Left eye inner corner',
            'Right eye inner corner', 
            'Right eye outer corner',
            'Nose tip',
            'Nose bottom',
            'Left mouth corner',
            'Right mouth corner'
        ]
        
        for i, (name, desc) in enumerate(zip(names, descriptions)):
            idx = indices[name]
            coord = keypoints_3d[i]
            f.write(f"# {desc} - MediaPipe index: {idx}\n")
            f.write(f"{coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n\n")
    
    print(f"\n✅ 座標已保存到: {output_file}")
    print(f"   共 {len(keypoints_3d)} 個關鍵點")


if __name__ == '__main__':
    print("=" * 70)
    print("從 MediaPipe Canonical Face Model 提取關鍵點")
    print("=" * 70)
    print()
    
    save_face_model()
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)

