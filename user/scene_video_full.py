from nuscenes.nuscenes import NuScenes
import cv2
import os

nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes-devkit/data/sets/nuscenes', verbose=True)

my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
# nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')

# 设置视频输出参数
output_filename = 'rendered_video.mp4'
output_fps = 10.0

# 获取渲染图像的路径
sample_data = nusc.get('sample_data', nusc.field2token('sample_data', 'channel', 'CAM_FRONT')[0])
# 更改工作目录到正确的位置
os.chdir('../data/nuscenes') 
image_folder = os.path.dirname(sample_data['filename'])
image_files = sorted(os.listdir(image_folder))

# 读取第一张图像，获取图像尺寸
sample_image = cv2.imread(os.path.join(image_folder, image_files[0]))
output_size = (sample_image.shape[1], sample_image.shape[0])

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_filename, fourcc, output_fps, output_size)

# 将图像写入视频
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)

# 释放资源
video_writer.release()

print(f"视频已保存为 {output_filename}")
