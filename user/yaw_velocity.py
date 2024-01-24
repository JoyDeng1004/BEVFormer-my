import datetime

from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pyquaternion import Quaternion

# 加载数据集
nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes-devkit/data/sets/nuscenes', verbose=True)

# 计算偏航速率和速度的函数
def calculate_yaw_rate_and_velocity(nusc):
    yaw_rates = []
    velocities = []
    tokens = []
    for sample in nusc.sample:
        sample_token = sample['token']
        tokens.append(sample_token)

        current_pose = sample['ego_pose']
        next_sample = nusc.get('sample', sample['next'])
        if next_sample is None:
            break
        next_pose = next_sample['ego_pose']

        # 计算偏航速率
        current_quat = Quaternion(current_pose['rotation'])
        next_quat = Quaternion(next_pose['rotation'])
        yaw_rate = Quaternion.angular_distance(current_quat, next_quat) / (next_pose['timestamp'] - current_pose['timestamp'])
        yaw_rates.append(yaw_rate)

        x = datetime.datetime.fromtimestamp(1532402927814384/ 1000000.0)
        y = datetime.datetime.fromtimestamp(1532402927888918/ 1000000.0)
        print(y-x)

        # 计算速度
        current_pos = np.array(current_pose['translation'])
        next_pos = np.array(next_pose['translation'])
        velocity = np.linalg.norm(next_pos - current_pos) / (next_pose['timestamp'] - current_pose['timestamp'])
        velocities.append(velocity)

    return tokens, yaw_rates, velocities

# 调用函数
sample_tokens, yaw_rates, velocities = calculate_yaw_rate_and_velocity(nusc)

# 输出结果
for token, yaw_rate, velocity in zip(sample_tokens, yaw_rates, velocities):
    print(f"Sample Token: {token}, Yaw Rate: {yaw_rate}, Velocity: {velocity}")

# 可视化和视频生成
# 这部分较复杂，需要根据具体的可视化需求来编写。可以使用matplotlib或OpenCV等库。


