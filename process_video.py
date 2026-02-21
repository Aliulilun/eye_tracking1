"""
影片視線估計處理腳本
Video Gaze Estimation Processor

功能：
- 讀取影片文件
- 逐幀進行視線估計
- 輸出帶視線標註的影片
- 導出每一幀的視線數據（CSV 格式）

作者: [你的名字]
日期: 2026-01
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml
import time
from tqdm import tqdm
import json

# 導入視線估計系統
from stages.stage1_face_detection import FaceDetector
from stages.stage2_head_pose import HeadPoseEstimator
from stages.stage3_normalization import ImageNormalizer
from stages.stage4_gaze_network import GazeEstimator
from stages.stage5_gaze_vector import GazeVectorConverter

from utils.camera_utils import get_default_camera_matrix
from utils.visualization import draw_gaze_arrow, draw_landmarks, add_text_overlay


class VideoGazeProcessor:
    """影片視線估計處理器"""
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化影片處理器
        
        Args:
            config_path: 配置文件路徑
        """
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 70)
        print("視線估計影片處理器初始化")
        print("Video Gaze Estimation Processor Initialization")
        print("=" * 70)
        
        # 初始化各個階段
        print("\n正在載入模型...")
        self.face_detector = FaceDetector(config=self.config['face_detection'])
        self.head_pose_estimator = HeadPoseEstimator(config=self.config['head_pose'])
        self.image_normalizer = ImageNormalizer(config=self.config['normalization'])
        self.gaze_estimator = GazeEstimator(config=self.config['model'])
        self.gaze_converter = GazeVectorConverter()
        
        print("✓ 模型載入完成！\n")
    
    def process_video(self, video_path, output_video_path=None, output_csv_path=None,
                     show_preview=False, skip_frames=0, max_frames=None,
                     draw_landmarks_flag=False):
        """
        處理影片文件
        
        Args:
            video_path: 輸入影片路徑
            output_video_path: 輸出影片路徑（如果為 None，不輸出影片）
            output_csv_path: 輸出 CSV 路徑（如果為 None，不輸出 CSV）
            show_preview: 是否顯示處理預覽
            skip_frames: 跳過的幀數（例如：skip_frames=2 表示每3幀處理1幀）
            max_frames: 最大處理幀數（用於測試）
            draw_landmarks_flag: 是否在輸出影片上繪製特徵點
        
        Returns:
            results_df: 包含所有幀視線數據的 DataFrame
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"影片文件不存在: {video_path}")
        
        print(f"\n處理影片: {video_path}")
        print("=" * 70)
        
        # 打開影片
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"無法打開影片: {video_path}")
        
        # 獲取影片資訊
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"影片資訊:")
        print(f"  - 解析度: {width} x {height}")
        print(f"  - FPS: {fps}")
        print(f"  - 總幀數: {total_frames}")
        print(f"  - 時長: {total_frames/fps:.2f} 秒")
        
        if skip_frames > 0:
            print(f"  - 跳幀設置: 每 {skip_frames + 1} 幀處理 1 幀")
            effective_frames = total_frames // (skip_frames + 1)
            print(f"  - 實際處理幀數: {effective_frames}")
        
        if max_frames:
            print(f"  - 最大處理幀數限制: {max_frames}")
        
        # 生成相機內參矩陣（基於影片解析度）
        camera_matrix = get_default_camera_matrix(width, height)
        print(f"\n使用默認相機內參矩陣:")
        print(f"  - 焦距: {camera_matrix[0, 0]:.2f} pixels")
        print(f"  - 主點: ({camera_matrix[0, 2]:.2f}, {camera_matrix[1, 2]:.2f})")
        
        # 準備輸出影片（如果需要）
        writer = None
        if output_video_path:
            output_video_path = Path(output_video_path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
            print(f"\n輸出影片: {output_video_path}")
        
        # 準備數據記錄
        results_data = []
        
        # 處理統計
        frame_idx = 0
        processed_count = 0
        failed_count = 0
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("開始處理影片...")
        print("=" * 70 + "\n")
        
        # 使用 tqdm 顯示進度
        pbar = tqdm(total=total_frames, desc="處理進度", unit="frames")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 更新進度條
                pbar.update(1)
                
                # 檢查是否達到最大幀數限制
                if max_frames and processed_count >= max_frames:
                    print(f"\n已達到最大處理幀數限制: {max_frames}")
                    break
                
                # 跳幀處理
                if frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    
                    # 如果需要輸出影片，寫入原始幀
                    if writer:
                        writer.write(frame)
                    
                    continue
                
                # 處理當前幀
                try:
                    result = self._process_frame(
                        frame, frame_idx, camera_matrix, 
                        draw_landmarks_flag=draw_landmarks_flag
                    )
                    
                    if result is not None:
                        results_data.append(result)
                        processed_count += 1
                        
                        # 繪製視線方向
                        annotated_frame = self._annotate_frame(
                            frame.copy(), result, draw_landmarks_flag
                        )
                        
                        # 顯示預覽
                        if show_preview:
                            cv2.imshow('Gaze Estimation', annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\n用戶中斷處理")
                                break
                        
                        # 寫入輸出影片
                        if writer:
                            writer.write(annotated_frame)
                    else:
                        failed_count += 1
                        
                        # 寫入原始幀（未檢測到人臉）
                        if writer:
                            writer.write(frame)
                
                except Exception as e:
                    # 處理失敗，記錄但繼續
                    failed_count += 1
                    if writer:
                        writer.write(frame)
                
                frame_idx += 1
        
        finally:
            # 清理資源
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # 處理統計
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("處理完成！")
        print("=" * 70)
        print(f"總幀數: {frame_idx}")
        print(f"成功處理: {processed_count} 幀")
        print(f"失敗/跳過: {failed_count} 幀")
        print(f"成功率: {processed_count/(processed_count+failed_count)*100:.1f}%")
        print(f"總耗時: {elapsed_time:.2f} 秒")
        print(f"平均速度: {frame_idx/elapsed_time:.2f} FPS")
        
        # 轉換為 DataFrame
        if results_data:
            results_df = pd.DataFrame(results_data)
            
            # 保存 CSV（如果需要）
            if output_csv_path:
                output_csv_path = Path(output_csv_path)
                output_csv_path.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(output_csv_path, index=False)
                print(f"\n✓ 視線數據已保存到: {output_csv_path}")
            
            # 顯示統計摘要
            print("\n" + "=" * 70)
            print("視線估計統計摘要:")
            print("=" * 70)
            print(f"Pitch (俯仰角):")
            print(f"  - 平均值: {results_df['gaze_pitch_deg'].mean():.2f}°")
            print(f"  - 標準差: {results_df['gaze_pitch_deg'].std():.2f}°")
            print(f"  - 範圍: [{results_df['gaze_pitch_deg'].min():.2f}°, {results_df['gaze_pitch_deg'].max():.2f}°]")
            
            print(f"\nYaw (偏航角):")
            print(f"  - 平均值: {results_df['gaze_yaw_deg'].mean():.2f}°")
            print(f"  - 標準差: {results_df['gaze_yaw_deg'].std():.2f}°")
            print(f"  - 範圍: [{results_df['gaze_yaw_deg'].min():.2f}°, {results_df['gaze_yaw_deg'].max():.2f}°]")
            
            print("=" * 70 + "\n")
            
            return results_df
        else:
            print("\n警告: 沒有成功處理任何幀！")
            return None
    
    def _process_frame(self, frame, frame_idx, camera_matrix, draw_landmarks_flag=False):
        """
        處理單個幀
        
        Args:
            frame: 輸入幀（BGR）
            frame_idx: 幀索引
            camera_matrix: 相機內參矩陣
            draw_landmarks_flag: 是否繪製特徵點
        
        Returns:
            result: 包含視線數據的字典，如果失敗返回 None
        """
        try:
            # 第一階段：人臉檢測
            face_data = self.face_detector.detect(frame)
            
            if face_data is None:
                return None
            
            # 第二階段：頭部姿態估計
            head_pose = self.head_pose_estimator.estimate(
                landmarks_2d=face_data['landmarks_2d_selected'],
                camera_matrix=camera_matrix
            )
            
            # 第三階段：圖像正規化
            normalized_data = self.image_normalizer.normalize(
                image=frame,
                landmarks=face_data['landmarks_2d_selected'],
                rotation_vector=head_pose['rvec'],
                translation_vector=head_pose['tvec'],
                camera_matrix=camera_matrix
            )
            
            # 第四階段：神經網絡推理
            gaze_output = self.gaze_estimator.predict(normalized_data['image'])
            
            # 第五階段：視線向量轉換
            gaze_vector_data = self.gaze_converter.pitchyaw_to_vector(
                pitch=gaze_output['pitch'],
                yaw=gaze_output['yaw']
            )
            
            # 組裝結果
            result = {
                'frame_idx': frame_idx,
                'timestamp_sec': frame_idx / 30.0,  # 假設 30 FPS，實際會在外部計算
                
                # 頭部姿態（度）
                'head_pitch_deg': head_pose['euler_angles']['pitch'],
                'head_yaw_deg': head_pose['euler_angles']['yaw'],
                'head_roll_deg': head_pose['euler_angles']['roll'],
                
                # 視線角度（弧度）
                'gaze_pitch_rad': gaze_output['pitch'],
                'gaze_yaw_rad': gaze_output['yaw'],
                
                # 視線角度（度）
                'gaze_pitch_deg': np.rad2deg(gaze_output['pitch']),
                'gaze_yaw_deg': np.rad2deg(gaze_output['yaw']),
                
                # 3D 視線向量
                'gaze_vector_x': gaze_vector_data['vector'][0],
                'gaze_vector_y': gaze_vector_data['vector'][1],
                'gaze_vector_z': gaze_vector_data['vector'][2],
                
                # 人臉位置
                'face_bbox_x': face_data['bbox'][0],
                'face_bbox_y': face_data['bbox'][1],
                'face_bbox_w': face_data['bbox'][2] - face_data['bbox'][0],
                'face_bbox_h': face_data['bbox'][3] - face_data['bbox'][1],
            }
            
            # 保存用於繪製的數據
            result['_landmarks'] = face_data['landmarks_2d_selected']
            result['_gaze_pitch'] = gaze_output['pitch']
            result['_gaze_yaw'] = gaze_output['yaw']
            
            return result
        
        except Exception as e:
            return None
    
    def _annotate_frame(self, frame, result, draw_landmarks_flag=False):
        """
        在幀上繪製視線方向和資訊
        
        Args:
            frame: 輸入幀
            result: 處理結果
            draw_landmarks_flag: 是否繪製特徵點
        
        Returns:
            annotated_frame: 標註後的幀
        """
        # 計算人臉中心
        face_center_x = int(result['face_bbox_x'] + result['face_bbox_w'] / 2)
        face_center_y = int(result['face_bbox_y'] + result['face_bbox_h'] / 2)
        
        # 繪製視線箭頭
        draw_gaze_arrow(
            frame, 
            (face_center_x, face_center_y),
            result['_gaze_pitch'],
            result['_gaze_yaw'],
            length=150,
            color=(0, 0, 255),
            thickness=3
        )
        
        # 繪製特徵點（可選）
        if draw_landmarks_flag and '_landmarks' in result:
            draw_landmarks(frame, result['_landmarks'], color=(0, 255, 0), radius=2)
        
        # 繪製人臉框
        cv2.rectangle(
            frame,
            (int(result['face_bbox_x']), int(result['face_bbox_y'])),
            (int(result['face_bbox_x'] + result['face_bbox_w']), 
             int(result['face_bbox_y'] + result['face_bbox_h'])),
            (0, 255, 0), 2
        )
        
        # 添加文字資訊
        info_text = [
            f"Frame: {result['frame_idx']}",
            f"Gaze Pitch: {result['gaze_pitch_deg']:>6.1f}deg",
            f"Gaze Yaw:   {result['gaze_yaw_deg']:>6.1f}deg",
        ]
        
        y_offset = 30
        for text in info_text:
            add_text_overlay(frame, text, position=(10, y_offset), 
                           font_scale=0.6, color=(0, 255, 0), thickness=2)
            y_offset += 25
        
        return frame


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='影片視線估計處理器 - Video Gaze Estimation Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 基本使用（處理影片並輸出結果）
  python process_video.py --input video.mp4 --output output.mp4 --csv data.csv
  
  # 僅導出數據，不輸出影片
  python process_video.py --input video.mp4 --csv data.csv
  
  # 顯示即時預覽
  python process_video.py --input video.mp4 --output output.mp4 --show-preview
  
  # 跳幀處理（提高速度）
  python process_video.py --input video.mp4 --skip-frames 2  # 每3幀處理1幀
  
  # 測試模式（只處理前 100 幀）
  python process_video.py --input video.mp4 --max-frames 100
  
  # 繪製面部特徵點
  python process_video.py --input video.mp4 --output output.mp4 --draw-landmarks
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='輸入影片路徑')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='輸出影片路徑（可選，如果不指定則不輸出影片）')
    parser.add_argument('--csv', '-c', type=str, default=None,
                       help='輸出 CSV 數據路徑（可選）')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路徑 (default: config.yaml)')
    parser.add_argument('--show-preview', action='store_true',
                       help='顯示處理預覽（按 q 退出）')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='跳過的幀數（例如：2 表示每3幀處理1幀，提高處理速度）')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大處理幀數（用於測試）')
    parser.add_argument('--draw-landmarks', action='store_true',
                       help='在輸出影片上繪製面部特徵點')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not Path(args.input).exists():
        print(f"錯誤: 輸入影片不存在: {args.input}")
        return 1
    
    # 如果沒有指定任何輸出，至少要有 CSV
    if args.output is None and args.csv is None:
        print("警告: 未指定輸出影片或 CSV，將僅輸出 CSV 到默認位置")
        args.csv = 'output/gaze_data.csv'
    
    try:
        # 初始化處理器
        processor = VideoGazeProcessor(config_path=args.config)
        
        # 處理影片
        results_df = processor.process_video(
            video_path=args.input,
            output_video_path=args.output,
            output_csv_path=args.csv,
            show_preview=args.show_preview,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            draw_landmarks_flag=args.draw_landmarks
        )
        
        if results_df is not None:
            print("\n✓ 影片處理完成！")
            return 0
        else:
            print("\n✗ 影片處理失敗！")
            return 1
    
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

