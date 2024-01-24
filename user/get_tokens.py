import torch
import argparse
import mmcv
import os
import torch
import warnings
import json
import time
import os.path as osp

from torchvision.transforms import functional as F
from PIL import Image

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes-devkit/data/sets/nuscenes', verbose=True)

def get_tokens(scene_index):
    # get the tokens of the first scene
    my_scene = nusc.scene[scene_index]
    print(my_scene)

    my_first_sample_token = my_scene['first_sample_token']
    my_last_sample_token = my_scene['last_sample_token']

    current_sample_token = my_first_sample_token

    my_tokens = []
    i = 0

    while current_sample_token != my_last_sample_token:
        current_sample_token = nusc.get('sample',current_sample_token)
        print(current_sample_token)
        current_sample_token_name = current_sample_token['token']
        my_tokens.append(current_sample_token_name)

        next_token = current_sample_token['next']
        current_sample_token = next_token

        i += 1

    my_tokens.append(my_last_sample_token)
    # print(my_tokens)
    # print(i)
    return my_tokens

# save tokens to JSON
def save_tokens_to_json(tokens, file_name):
    # dictionary
    tokens_dict = {str(i): token for i, token in enumerate(tokens)}
    # write in
    with open(file_name, 'w') as f:
        json.dump(tokens_dict, f, indent=4)

my_scene_index = 0
my_tokens = get_tokens(my_scene_index)
# print(my_tokens)
output_file_name = 'my_tokens.json'
# save tokens
save_tokens_to_json(my_tokens, output_file_name)

###================================================================###

# import torch
# from mmcv import Config
# from mmdet3d.models import build_model
# from mmdet3d.models.detectors import BEVFormer
# from mmdet3d.apis import init_model
# from mmdet3d.datasets import build_dataset, build_dataloader

# # 假设我们已经知道了TemporalSelfAttention和SpatialCrossAttention的实现位置
# # 从相应的模块导入这些类
# from mmdet3d.models.utils.transformer import TemporalSelfAttention, SpatialCrossAttention

# # 定义一个全局变量来存储挂钩的输出
# global_features = {}

# # 定义挂钩函数
# def forward_hook(module, input, output):
#     global global_features
#     global_features[module] = output

# # 加载配置文件和模型权重
# cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')
# # 如果有预训练模型权重，可以在此处指定
# model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# # 将模型设置为评估模式
# model.eval()

# # 假设我们要挂钩的模块是encoder中的TemporalSelfAttention和SpatialCrossAttention
# for name, module in model.named_modules():
#     if isinstance(module, TemporalSelfAttention):
#         module.register_forward_hook(forward_hook)
#     elif isinstance(module, SpatialCrossAttention):
#         module.register_forward_hook(forward_hook)

# # 选择BEV查询的层并添加挂钩
# # 假设BEV查询在模型的某个特定层中
# model.bev_embedding.register_forward_hook(forward_hook)

# # 准备数据加载器
# dataset = build_dataset(cfg.data.test)
# data_loader = build_dataloader(
#     dataset,
#     samples_per_gpu=1,
#     workers_per_gpu=cfg.data.workers_per_gpu,
#     dist=False,
#     shuffle=False)

# # 使用DataLoader迭代特定的样本
# for i, data in enumerate(data_loader):
#     # 只对我们感兴趣的样本进行处理
#     if data['sample_idx'].item() in my_tokens:
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)

# # 检查挂钩的输出
# for module, feature in global_features.items():
#     print(f"Feature from {module}: {feature.shape}")

# ###================================================================###

# # 加载你的配置文件和预训练模型
# cfg = mmcv.Config.fromfile('configs/bevformer/bevformer_something.py')
# model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# checkpoint = load_checkpoint(model, 'checkpoints/bevformer_something.pth', map_location='cpu')

# # 如果你的模型在GPU上
# model = model.cuda()
# model.eval()

# # 注册钩子来保存特定层的输出
# features = {}
# def save_features(name):
#     def hook(module, input, output):
#         features[name] = output
#     return hook

# # 假设我们要保存 temporal self attention 和 spatial cross attention 的输出
# # 你需要知道这些模块的名称或路径
# model.module_name.register_forward_hook(save_features('temporal_self_attention'))
# model.another_module_name.register_forward_hook(save_features('spatial_cross_attention'))

# # 准备输入数据
# # 这里需要使用正确的数据加载和预处理方式来创建输入数据
# # input_data = ...

# # 将输入数据转移到正确的设备上，并进行前向传播
# # 如果输入数据是字典或列表，可能需要使用collate函数
# # 如果使用多GPU，可能需要使用scatter函数
# input_data = collate([input_data], samples_per_gpu=1)
# if next(model.parameters()).is_cuda:
#     # scatter to specified GPU
#     input_data = scatter(input_data, [next(model.parameters()).get_device()])[0]

# with torch.no_grad():
#     output = model(return_loss=False, rescale=True, **input_data)

# # 检查保存的特征
# print(features['temporal_self_attention'])
# print(features['spatial_cross_attention'])

'''
class BEVFormerModified(nn.Module):
    def __init__(self, ...):  # 添加适当的参数
        super().__init__()
        # 初始化 BEVFormer 的各个层
        ...

    def forward(self, x):
        # 假设我们想保存第一个 transformer 层的输出
        x1 = self.transformer_layer1(x)
        self.intermediate_output = x1.detach()  # 保存中间输出

        # 继续前向传播
        ...
        return output

# 然后在您的主函数中
if __name__ == '__main__':
    model = BEVFormerModified(...)  # 使用适当的参数初始化模型
    ...
    output = model(input_data)  # 执行前向传播
    # 现在可以访问 model.intermediate_output 来获取保存的中间层输出
'''

'''
class BEVFormerModified(BEVFormer):
    def __init__(self, ...):  # 添加适当的参数
        super().__init__(...)
        # 初始化 BEVFormer 的其他组件
        ...

    def forward(self, x):
        # 前向传播逻辑...
        # 假设 intermediate_layer 是我们想要保存输出的层
        intermediate_output = self.intermediate_layer(x)
        # 保存中间输出
        self.saved_output = intermediate_output.detach()
        # 继续前向传播
        output = ... # 原始的前向传播输出
        return output

# 在 test.py 的 main 函数中
if __name__ == '__main__':
    ...
    model = BEVFormerModified(...)  # 使用您修改后的模型类
    ...
    if not distributed:
        ...
    else:
        ...
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
        # 在这里，您可以访问 model.saved_output 来获取保存的中间层输出

'''


'''
# 加载模型配置
config_path = '../work_dirs/bevformer_base/latest.pth'
config = load_config(config_path)

# 创建并加载BEVFormer模型
model = BEVFormer(config)
model.load_state_dict(torch.load('path_to_your_pretrained_model.pth'))
model.eval()


# 获取第一个record的信息
my_scene = nusc.scene[0]

# 提取场景信息
scene_id = my_scene['name']
frame_count = my_scene['nbr_samples']
first_sample_token = my_scene['first_sample_token']

# 获取第一个sample的token
sample_token = first_sample_token

# 遍历每个时间步
for _ in range(frame_count):
    # 使用nuscenes库获取sample信息
    sample = nusc.get('sample', sample_token)

    # 获取图像路径
    image_path = nusc.get_sample_data_path(sample['data']['CAM_FRONT'])

    # 加载图像，并将其转换为模型所需的输入格式
    image = Image.open(image_path).convert('RGB')
    image = F.to_tensor(image).unsqueeze(0)

    # 将图像输入到BEVFormer模型中
    with torch.no_grad():
        output = model.encoder(image)

    # 输出BEV query
    print(f"Scene ID: {scene_id}, Sample Token: {sample_token}")
    print("BEV Query:\n", output)

    # 获取下一个sample的token
    sample_token = sample['next']

    # 如果没有下一个sample，退出循环
    if sample_token == '':
        break
'''