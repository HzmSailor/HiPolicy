import cv2
import numpy as np
import json
import os
import math
import random
from pathlib import Path
import pandas as pd

image_dir = "/DATA/disk1/eval_video/dp/shoe_place_D435_100_seed0/observer/test_38"
end_pose_dir = "/DATA/disk1/eval_video/dp/shoe_place_D435_100_seed0/endpose/endpose_coords_38.csv"
output_video = "/DATA/disk1/eval_video/dp/shoe_place_D435_100_seed0/output_video_observer_master.mp4"
output_points_file = "selected_points.txt"

end_pose = pd.read_csv(end_pose_dir)
print(end_pose.columns)
end_pose = end_pose.to_numpy()
print(end_pose.shape)

left_coords = end_pose[:, 1:4]
right_coords = end_pose[:, 4:7]

print(left_coords.shape)
print(right_coords.shape)

# 相机内参
fovy = 37
camera_w = 320
camera_h = 240
focal_length = (camera_h / 2) / math.tan(math.radians(fovy / 2))
intrinsic_matrix = np.array([
    [focal_length, 0, camera_w / 2],
    [0, focal_length, camera_h / 2],
    [0, 0, 1]
], dtype=np.float32)

# 视频参数
fps = 2
frame_size = (camera_w, camera_h)

# 轨迹长度和样式
trajectory_lengths = [2, 3, 7]
left_colors = [(0, 100, 0), (0, 128, 0), (0, 255, 127)]   # 深绿、标准绿、春绿色
right_colors = [(0, 0, 139), (0, 0, 255), (0, 255, 255)]  # 深蓝、亮蓝、青色

offsets = [-5, 0, 5]  # x轴偏移像素

# # 交互式标定
# def select_points(frame_idx, img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"无法读取图像: {img_path}")
#         return None, None
    
#     img = cv2.resize(img, (camera_w, camera_h))
#     temp_points = []
    
#     def mouse_callback(event, u, v, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             param['points'].append((u, v))
#             print(f"点击坐标: ({u}, {v})")
#             cv2.circle(param['img'], (u, v), 5, (0, 255, 0), -1)
#             cv2.imshow("Image", param['img'])
    
#     print(f"\n帧 {frame_idx}: 请在图像上点击左侧机械臂末端（红点），然后点击右侧机械臂末端（蓝点）")
#     cv2.namedWindow("Image")
#     cv2.setMouseCallback("Image", mouse_callback, {'img': img, 'points': temp_points})
    
#     while len(temp_points) < 2:
#         cv2.imshow("Image", img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # 按ESC退出
#             break
    
#     cv2.imwrite(f"selected_points_frame_{frame_idx}.png", img)
#     cv2.destroyAllWindows()
    
#     if len(temp_points) == 2:
#         left_point = temp_points[0]
#         right_point = temp_points[1]
#         return left_point, right_point
#     return None, None

# # 随机选择5帧进行标定
# num_frames = len(image_files)
# calibration_frames = sorted(random.sample(range(num_frames), 5))
# print(f"选择的标定帧: {calibration_frames}")

# # 收集3D-2D对应点
# object_points = []
# image_points = []
# calibration_data = []

# for frame_idx in calibration_frames:
#     img_path = os.path.join(image_dir, f"{frame_idx}.png")
#     left_pose_path = os.path.join(left_pose_dir, f"{frame_idx}.json")
#     right_pose_path = os.path.join(right_pose_dir, f"{frame_idx}.json")
    
#     # 获取3D坐标
#     with open(left_pose_path, 'r') as f:
#         left_pose = json.load(f)
#     with open(right_pose_path, 'r') as f:
#         right_pose = json.load(f)
    
#     left_3d = [left_pose['x'], left_pose['y'], left_pose['z']]
#     right_3d = [right_pose['x'], right_pose['y'], right_pose['z']]
    
#     # 获取2D坐标
#     left_2d, right_2d = select_points(frame_idx, img_path)
#     if left_2d is None or right_2d is None:
#         print(f"帧 {frame_idx}: 标定失败，跳过")
#         continue
    
#     object_points.extend([left_3d, right_3d])
#     image_points.extend([left_2d, right_2d])
#     calibration_data.append((frame_idx, left_2d[0], left_2d[1], right_2d[0], right_2d[1]))
#     print(f"帧 {frame_idx}: 左侧3D={left_3d}, 左侧2D={left_2d}, 右侧3D={right_3d}, 右侧2D={right_2d}")

# # 保存标定点
# with open(output_points_file, 'w') as f:
#     for frame_idx, left_u, left_v, right_u, right_v in calibration_data:
#         f.write(f"Frame {frame_idx}: Left=({left_u}, {left_v}), Right=({right_u}, {right_v})\n")
# print(f"标定点已保存到 {output_points_file}")

# # 转换为numpy数组
# object_points = np.array(object_points, dtype=np.float32)
# image_points = np.array(image_points, dtype=np.float32)

# # 检查标定点数量
# if len(object_points) < 6:
#     print("标定点不足（至少需要6个点），无法计算外参")
#     exit()

# # 计算外参矩阵
# ret, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsic_matrix, None)
# if not ret:
#     print("外参计算失败")
#     exit()
# R, _ = cv2.Rodrigues(rvec)
# extrinsic_matrix = np.hstack((R, tvec))
# extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
# print("外参矩阵:\n", extrinsic_matrix)
# print("RANSAC内点数:", len(inliers))

extrinsic_matrix = np.array([
    [-0.56147192,  0.81839839,  0.12236567,  0.06534784],
    [ 0.35113617,  0.36953459, -0.8603183,   0.87536854],
    [-0.74930146, -0.44007755, -0.49485257,  2.67977485],
    [ 0.0, 0.0, 0.0, 1.0]
])

# 检查外参正交性
R = extrinsic_matrix[:3, :3]
print("旋转矩阵正交性:", np.allclose(R.T @ R, np.eye(3), atol=1e-6))

step_indices = end_pose[:, 0].astype(int)
num_frames = len(step_indices)

# 检查图像尺寸
img_path = os.path.join(image_dir, f"observer_step_{step_indices[0]}.png")
img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图像用于尺寸检查: {img_path}，将跳过尺寸打印")
else:
    print(f"原始图像尺寸: {img.shape}")

# 初始化视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

def project_3d_to_2d(point_3d, intrinsic, extrinsic):
    point_3d_hom = np.array([point_3d[0], point_3d[1], point_3d[2], 1], dtype=np.float32)
    point_camera = extrinsic @ point_3d_hom
    point_2d_hom = intrinsic @ point_camera[:3]
    if point_2d_hom[2] > 0:
        point_2d = point_2d_hom[:2] / point_2d_hom[2]
        return point_2d
    return None

# 处理帧
for i in range(num_frames):
    img_path = os.path.join(image_dir, f"observer_step_{step_indices[i]}.png")
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"无法读取图像 {img_path}")
        continue

    frame = cv2.resize(frame, frame_size)

    # 为每种轨迹长度绘制轨迹
    for length_idx, length in enumerate(trajectory_lengths):
        # 收集当前窗口的折线点
        left_polyline_points = []
        right_polyline_points = []
        for j in range(i, min(i + length, num_frames)):
            left_point_3d = left_coords[j]
            right_point_3d = right_coords[j]

            left_point_2d = project_3d_to_2d(left_point_3d, intrinsic_matrix, extrinsic_matrix)
            right_point_2d = project_3d_to_2d(right_point_3d, intrinsic_matrix, extrinsic_matrix)

            if i < 5:
                print(f"帧 {i}, 点 {j}, 长度 {length}: 左侧3D={left_point_3d}, 左侧2D={left_point_2d}, 右侧3D={right_point_3d}, 右侧2D={right_point_2d}")

            # 绘制左侧轨迹点并收集折线点
            if left_point_2d is not None:
                x_left, y_left = int(left_point_2d[0] + offsets[length_idx]), int(left_point_2d[1])
                if 0 <= x_left < camera_w and 0 <= y_left < camera_h:
                    left_polyline_points.append((x_left, y_left))
                    cv2.circle(frame, (x_left, y_left), 1, left_colors[length_idx], -1)
            
            # 绘制右侧轨迹点并收集折线点
            if right_point_2d is not None:
                x_right, y_right = int(right_point_2d[0] + offsets[length_idx]), int(right_point_2d[1])
                if 0 <= x_right < camera_w and 0 <= y_right < camera_h:
                    right_polyline_points.append((x_right, y_right))
                    cv2.circle(frame, (x_right, y_right), 1, right_colors[length_idx], -1)

        # 将每种颜色的点连成线
        if len(left_polyline_points) >= 2:
            left_pts = np.array(left_polyline_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [left_pts], isClosed=False, color=left_colors[length_idx], thickness=1)
        if len(right_polyline_points) >= 2:
            right_pts = np.array(right_polyline_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [right_pts], isClosed=False, color=right_colors[length_idx], thickness=1)

    if i < 5:
        cv2.imwrite(f"test_frame_{i}.png", frame)

    out.write(frame)

out.release()
print(f"视频已保存为 {output_video}")