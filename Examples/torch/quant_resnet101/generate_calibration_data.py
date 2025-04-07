import os, torch
import random
import shutil
import numpy as np
import paramiko

from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torch.utils.data import Dataset
import torch.utils.data as torch_data


from Examples.common import image_net_config, config_param
from Examples.torch.utils import image_net_data_loader


# 新增一个自定义的Dataset类来处理校准图像
class ImageNetCalibrationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        
        for filename in os.listdir(root):
            file_extension = os.path.splitext(filename)[1]
            if file_extension.lower() in allowed_extensions:
                path = os.path.join(root, filename)
                self.samples.append((path, 0))  # 使用0作为占位符标签

        # print(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    
def save_tensor_as_raw(tensor, filename):
    # 将tensor转换为numpy数组
    np_array = tensor.cpu().numpy()
    # 将numpy数组保存为raw文件
    np_array.tofile(filename)
 
 
def read_raw_file(file_path, shape=None):
    """
    读取 raw 文件并将其转换为 PyTorch 张量。

    Args:
        file_path (str): raw 文件的路径。
        shape (tuple, optional): 期望的张量形状。如果为 None，则返回一维张量。

    Returns:
        torch.Tensor: 包含 raw 文件数据的 PyTorch 张量。
    """
    # 读取 raw 文件
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # 转换为 PyTorch 张量
    tensor = torch.from_numpy(data)
    
    # 如果指定了形状，则重塑张量
    if shape is not None:
        tensor = tensor.reshape(shape)
    
    return tensor
   
    
def copy_imaggnet_to_dir(source_dir, target_dir):
    # 源数据集目录
    source_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset/train_mini_200'

    # 目标目录(用于保存选中的图片)
    target_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images'

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录中的所有子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        if os.path.isdir(subdir_path):
            # 检查是否存在 "images" 子目录
            images_dir = os.path.join(subdir_path, 'images')
            if os.path.isdir(images_dir):
                # 获取 "images" 子目录中所有图片文件
                images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    # 随机选择一张图片
                    selected_image = random.choice(images)
                    print(selected_image)
                    source_path = os.path.join(images_dir, selected_image)
                    target_path = os.path.join(target_dir, f"{subdir}_{selected_image}")
                    
                    # 复制选中的图片到目标目录
                    shutil.copy2(source_path, target_path)

    print(f"已完成。从每个子目录的 'images' 文件夹中随机选择了一张图片，并保存到 {target_dir}")


def copy_imaggnet_to_sub_dir(source_dir, target_dir):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录及其所有子目录
    for root, dirs, files in os.walk(source_dir):
        # 过滤出图片文件
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # 随机选择一张图片
            selected_image = random.choice(image_files)
            
            # 获取源文件的完整路径
            source_path = os.path.join(root, selected_image)
            
            # 计算目标路径，将图片移动到上一级目录
            relative_path = os.path.relpath(root, source_dir)
            parent_dir = os.path.dirname(relative_path)  # 获取上一级目录
            target_path = os.path.join(target_dir, parent_dir, selected_image)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f"Copied: {source_path} -> {target_path}")

    print(f"复制完成。从每个子目录中随机选择了一张图片，并复制到上一级目录在 {target_dir} 中。")


def after_preprocess_generate_calibration_data(calibration_data_dir):
    """_summary_

    Args:
        calibration_data_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    normalize = transforms.Normalize(mean=image_net_config.dataset['images_mean'],
                                         std=image_net_config.dataset['images_std'])
    
    val_transforms = transforms.Compose([
            transforms.Resize(image_net_config.dataset["image_size"]+ 24),
            transforms.CenterCrop(image_net_config.dataset["image_size"]),
            transforms.ToTensor(),
            normalize])
    
    data_set = ImageNetCalibrationDataset(
            root=calibration_data_dir,
            transform=val_transforms
    )
    
    _data_loader = torch_data.DataLoader(
        data_set, batch_size=1, shuffle=False,
        num_workers=16, pin_memory=True)
    
    return _data_loader


def process_calibration_image(img_output_dir, raw_output_dir):
    img_output_dir = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images_sub_dir"
    raw_output_dir = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_raw_sub_dir"
    
    # 这个目录存放的都是jpg格式的校准数据图片
    img_data_path = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images/calibration_image_datas"
    
    data_loader = after_preprocess_generate_calibration_data(img_data_path)
    for i, input_data in enumerate(data_loader):
        
        # 获取图片所在的每个子目录
        each_dir = os.listdir(img_output_dir)
        
        # 每个图片所在的目录路径
        dir_path = os.path.join(img_output_dir, each_dir[i])
        
        # 便利目录下的图片
        for each_img in os.listdir(dir_path):
            
            # 保存到raw 格式下的子目录          
            raw_dir = os.path.join(raw_output_dir, each_dir[i])
            os.makedirs(raw_dir, exist_ok=True)
            # img_name = os.path.splitext(each_img)[0]
            
            # 这里的校准数据的raw文件名字，要和输入的名字保持一致，否则量化会报错呢
            raw_file_path = os.path.join(raw_dir, "input.raw")
            
            save_tensor_as_raw(input_data, raw_file_path)
            
            print(f"Saved tensor to {raw_file_path}")
            
            
def process_validation_dataset(val_data_path, val_raw_data):
    
    normalize = transforms.Normalize(mean=image_net_config.dataset['images_mean'],
                                         std=image_net_config.dataset['images_std'])
    
    val_transforms = transforms.Compose([
            transforms.Resize(image_net_config.dataset["image_size"]+ 24),
            transforms.CenterCrop(image_net_config.dataset["image_size"]),
            transforms.ToTensor(),
            normalize])
    
    
    data_set = image_net_data_loader.ImageFolder(
            root=val_data_path,
            transform=val_transforms
    )
    
    _data_loader = torch_data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    txt_file_path = os.path.join(val_raw_data, '/mnt/share_disk/bruce_trie/workspace/perception_quanti/demo_18/index_label.txt')

    with open(txt_file_path, 'w') as txt_file:

        for index, (input_data, input_label) in enumerate(_data_loader):
            
            val_raw_file_path = os.path.join(val_raw_data, f"{index}_{input_label.item()}.raw")
            save_tensor_as_raw(input_data, val_raw_file_path)
            print(f"Saved tensor to {val_raw_file_path}")
            
            txt_file.write(f"{index}:{input_label.item()}\n")
        
        
    
    # return _data_loader
    

def generate_validation_file(val_raw_data_path, output_file):
    # 确保输入目录存在
    if not os.path.exists(val_raw_data_path):
        print(f"错误: 目录 '{val_raw_data_path}' 不存在")
        return

    # 获取目录中的所有文件
    files = os.listdir(val_raw_data_path)

    # 过滤出图片文件 (这里假设图片文件扩展名为 .raw)
    image_files = [f for f in files if f.lower().endswith('.raw')]

    # 写入文件
    with open(output_file, 'w') as f:
        for image_file in image_files:
            full_path = os.path.join(val_raw_data_path, image_file)
            f.write(f"input:={full_path}\n")

    print(f"已将 {len(image_files)} 个图片文件路径写入 {output_file}")


         
           
if __name__ == "__main__":
    
    
    # 1 拷贝图片
    # source_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset/train_mini_200'

    # # 目标目录(用于保存选中的图片)
    # target_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images_sub_dir'
    # copy_imaggnet_to_sub_dir(source_dir, target_dir)
    
    
    # 2 处理校准的图片数据    
    # process_calibration_image()
    
    # 3 处理量化验证的图片数据
    val_data_path = "/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset/val"
    val_raw_data = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_validate_raw_dir"
    # os.makedirs(val_raw_data, exist_ok=True)
    # process_validation_dataset(val_data_path, val_raw_data)
    
    # 4 创建验证数据集txt文件
    txt_path = "/mnt/share_disk/bruce_trie/workspace/perception_quanti/demo_18/demo_18_int8/input_list.txt"
    # generate_validation_file(val_raw_data, txt_path)

    # 5 验证精度
    quant_demo_18_output_path = "/mnt/share_disk/bruce_trie/workspace/outputs/demo_18_output"
    