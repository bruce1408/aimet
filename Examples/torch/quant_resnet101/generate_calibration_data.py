import os
import random
import shutil
import os

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
            print(filename)
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


if __name__ == "__main__":
    
    # 1 拷贝图片
    # source_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset/train_mini_200'

    # # 目标目录(用于保存选中的图片)
    # target_dir = '/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images_sub_dir'
    # copy_imaggnet_to_sub_dir(source_dir, target_dir)
    
    # 2 处理图片    
    img_output_dir = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images_sub_dir"
    raw_output_dir = "/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_raw_sub_dir"
    data_loader = after_preprocess_generate_calibration_data("/mnt/share_disk/bruce_trie/workspace/outputs/classify_calibration_images/calibration_image_datas")
    for i, input_data in enumerate(data_loader):
        each_dir = os.listdir(img_output_dir)
        dir_path = os.path.join(img_output_dir, each_dir[i])
        for each_img in os.listdir(dir_path):
            img_path = os.path.join(dir_path, each_img)
            
            raw_dir = os.path.join(raw_output_dir, each_dir[i])
            os.makedirs(raw_dir, exist_ok=True)
            img_name = os.path.splitext(each_img)[0]
            
            # 这里的校准数据的raw文件名字，要和输入的名字保持一致，否则量化会报错呢
            raw_file_path = os.path.join(raw_dir, "input.raw")
            save_tensor_as_raw(input_data, raw_file_path)
            print(f"Saved tensor to {raw_file_path}")