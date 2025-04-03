import torch
import onnx, os
import torch,argparse
import numpy as np
from torchvision import transforms
import onnxruntime as ort
from typing import Tuple
from Examples.common import config_param
from Examples.common import image_net_config
from Examples.torch.utils.image_net_data_loader import ImageFolder
from spectrautils import logging_utils,print_utils,time_utils

os.environ["CUDA_VISIBLE_DEVICES"]=config_param.cuda_ids


parser = argparse.ArgumentParser(description='ResNet18 onnx model evaluate on ImageNet dataset')

parser.add_argument('--dataset_dir', 
                    type=str,
                    default=config_param.imagenet_dir,
                    help="Path to a directory containing ImageNet dataset.\n\
                            This folder should conatin at least 2 subfolders:\n\
                            'train': for training dataset and 'val': for validation dataset")

parser.add_argument('--use_cuda', 
                    type=bool,
                    default=True,
                    help='Add this flag to run the test on GPU.')

parser.add_argument('--logdir', 
                    type=str,
                    default=config_param.aimet_log_dir,
                    help="Path to a directory for logging. Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

parser.add_argument('--log_prefix', 
                   type=str,
                   default="resnet18_onnx_acc",
                   help='Custom prefix for log files')


_config = parser.parse_args()


logger_manager = logging_utils.AsyncLoggerManager(work_dir=_config.logdir, name_prefix=_config.log_prefix)
logger = logger_manager.logger

def prepare_data(dataset_dir, num_samples_per_class=1000):
    image_size = 224
    
    normalize = transforms.Normalize(
        mean=image_net_config.dataset['images_mean'],
        std=image_net_config.dataset['images_std'])

    val_transforms = transforms.Compose([
                transforms.Resize(image_size + 24),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize])

    data_set = ImageFolder(
                root=os.path.join(dataset_dir, 'val'),
                transform=val_transforms,
                num_samples_per_class=num_samples_per_class)
    
    return data_set

@time_utils.time_it
def evaluate_model(session: ort.InferenceSession, dataset: ImageFolder) -> Tuple[int, float]:
    """评估ONNX模型"""
    correct = 0
    total = 0
    input_name = session.get_inputs()[0].name
    
    for batch_idx, (inputs, labels) in enumerate(dataset, 1):
        input_data = inputs.unsqueeze(0).numpy()
        outputs = session.run(None, {input_name: input_data})
        _, predicted = torch.max(torch.tensor(outputs[0]), 1)
        
        total += 1
        correct += (predicted.item() == labels) if isinstance(labels, int) else (predicted == labels).sum().item()
        
        # 每200个样本或最后一批打印进度
        if batch_idx % 200 == 0 or batch_idx == len(dataset):
            accuracy = 100 * correct / total
            progress = 100 * batch_idx / len(dataset)
            logger.info(f"评估进度: {progress:.1f}% ({total}/{len(dataset)}) | 当前批次准确率: {accuracy:.2f}%") 
            # logger.info(f"🔄 进度: {total:5d}/{len(data_set)} | ✅ 准确率: {accuracy:6.2f}%")
    return total, 100 * correct / total


def main():
    # 准备数据和模型
    dataset = prepare_data(_config.dataset_dir)
    providers = ['CUDAExecutionProvider'] if _config.use_cuda else ['CPUExecutionProvider']
    
    # 加载模型
    session = ort.InferenceSession(config_param.onnx_resnet18_path, providers=providers)
    
    # 评估模型
    total_samples, final_accuracy = evaluate_model(session, dataset)
    
    logger.info(f'Accuracy of the ONNX model on the {total_samples} test images: {final_accuracy:.2f}%')
    


if __name__ == "__main__":
    main()