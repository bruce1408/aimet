
# # AutoQuant
# 
# This notebook shows a working code example of how to use AIMET AutoQuant feature.
# 
# AIMET offers a suite of neural network post-training quantization (PTQ) techniques that can be applied in succession. However, the process of finding the right combination and sequence of techniques to apply is time-consuming and requires careful analysis, which can be challenging especially for non-expert users. We instead recommend AutoQuant to save time and effort.
# 
# AutoQuant is an API that applies various PTQ techniques in AIMET automatically based on analyzing the model and best-known heuristics. In AutoQuant, users specify the amount of tolerable accuracy drop, and AutoQuant will apply PTQ techniques cumulatively until the target accuracy is satisfied.
# 
# 
# #### Overall flow
# This notebook covers the following
# 1. Define constants and helper functions
# 2. Load a pretrained FP32 model
# 3. Run AutoQuant
# 
# #### What this notebook is not
# This notebook is not designed to show state-of-the-art AutoQuant results. For example, it uses a relatively quantization-friendly model like Resnet18. Also, some optimization parameters are deliberately chosen to have the notebook execute more quickly.
# 


# ---
# ## Dataset
# 
# This notebook relies on the ImageNet dataset for the task of image classification. If you already have a version of the dataset readily available, please use that. Else, please download the dataset from appropriate location (e.g. https://image-net.org/challenges/LSVRC/2012/index.php#).
# 
# **Note1**: The ImageNet dataset typically has the following characteristics and the dataloader provided in this example notebook rely on these
# - Subfolders 'train' for the training samples and 'val' for the validation samples. Please see the [pytorch dataset description](https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html) for more details.
# - A subdirectory per class, and a file per each image sample
# 
# **Note2**: To speed up the execution of this notebook, you may use a reduced subset of the ImageNet dataset. E.g. the entire ILSVRC2012 dataset has 1000 classes, 1000 training samples per class and 50 validation samples per class. But for the purpose of running this notebook, you could perhaps reduce the dataset to say 2 samples per class. This exercise is left upto the reader and is not necessary.
# 
# Edit the cell below and specify the directory where the downloaded ImageNet dataset is saved.

import os
from Examples.common import config_param
from torchvision import transforms, datasets
os.environ["CUDA_VISIBLE_DEVICES"]=config_param.cuda_ids

DATASET_DIR = '/mnt/share_disk/bruce_trie/outputs/imagenet_dataset'         

val_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

imagenet_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, 'val'), transform=val_transforms)


# ## 1. Define Constants and Helper functions
# 
# In this section the constants and helper functions needed to run this eaxmple are defined.
# 
# - **EVAL_DATASET_SIZE** A typical value is 5000. In this notebook, this value has been set to 500 for faster execution.
# - **CALIBRATION_DATASET_SIZE** A typical value is 2000. In this notebook, this value has been set to 200 for faster execution.
# 
# 
# The helper function **_create_sampled_data_loader()** returns a DataLoader based on the dataset and the number of samples provided.


import random
from typing import Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from aimet_torch.utils import in_eval_mode, get_device

EVAL_DATASET_SIZE = 500
CALIBRATION_DATASET_SIZE = 200

_datasets = {}

def _create_sampled_data_loader(dataset, num_samples):
    if num_samples not in _datasets:
        indices = random.sample(range(len(dataset)), num_samples)
        _datasets[num_samples] = Subset(dataset, indices)
    return DataLoader(_datasets[num_samples], batch_size=32)


def eval_callback(model: torch.nn.Module, num_samples: Optional[int] = None) -> float:
    if num_samples is None:
        num_samples = EVAL_DATASET_SIZE

    data_loader = _create_sampled_data_loader(imagenet_dataset, num_samples)
    device = get_device(model)
    
    correct = 0
    with in_eval_mode(model), torch.no_grad():
        for image, label in tqdm(data_loader):
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            top1 = logits.topk(k=1).indices
            correct += (top1 == label.view_as(top1)).sum()

    return int(correct) / num_samples


# ## 2. Load a pretrained FP32 model
# For this example, we are going to load a pretrained resnet18 model from torchvision. Similarly, you can load any pretrained PyTorch model instead.


from torchvision.models import resnet18

model = resnet18(pretrained=True).eval()

if torch.cuda.is_available():
    model.to(torch.device('cuda'))

accuracy = eval_callback(model)
print(f'- FP32 accuracy: {accuracy}')


# ## 3. Run AutoQuant
# ### Create AutoQuant Object
# 
# The AutoQuant feature utilizes an unlabeled dataset to achieve quantization. The class **UnlabeledDatasetWrapper** creates an unlabeled Dataset object from a labeled Dataset. 


from aimet_torch.auto_quant import AutoQuant

class UnlabeledDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        images, _ = self._dataset[index]
        return images


unlabeled_imagenet_dataset = UnlabeledDatasetWrapper(imagenet_dataset)
unlabeled_imagenet_data_loader = _create_sampled_data_loader(unlabeled_imagenet_dataset,
                                                             CALIBRATION_DATASET_SIZE)

dummy_input = torch.randn((1, 3, 224, 224)).to(get_device(model))

auto_quant = AutoQuant(model,
                       dummy_input=dummy_input,
                       data_loader=unlabeled_imagenet_data_loader,
                       eval_callback=eval_callback)


# ### Run AutoQuant Inference
# This step runs AutoQuant inference. AutoQuant inference will run evaluation using the **eval_callback** with the vanilla quantized model without applying PTQ techniques. This will be useful for measuring the baseline evaluation score before running AutoQuant optimization.


sim, initial_accuracy = auto_quant.run_inference()
print(f"- Quantized Accuracy (before optimization): {initial_accuracy}")


# ### Set AdaRound Parameters (optional)
# AutoQuant uses a set of predefined default parameters for AdaRound.
# These values were determined empirically and work well with the common models.
# However, if necessary, you can also use your custom parameters for Adaround.
# In this notebook, we will use very small AdaRound parameters for faster execution.


from aimet_torch.adaround.adaround_weight import AdaroundParameters

ADAROUND_DATASET_SIZE = 200
adaround_data_loader = _create_sampled_data_loader(unlabeled_imagenet_dataset, ADAROUND_DATASET_SIZE)
adaround_params = AdaroundParameters(adaround_data_loader, num_batches=len(adaround_data_loader), default_num_iterations=2000)
auto_quant.set_adaround_params(adaround_params)


# ### Run AutoQuant Optimization
# This step runs AutoQuant optimization, which returns the best possible quantized model, corresponding evaluation score and the path to the encoding file.
# The **allowed_accuracy_drop** parameter indicates the tolerable amount of accuracy drop. AutoQuant applies a series of quantization features until the target accuracy (FP32 accuracy - allowed accuracy drop) is satisfied. When the target accuracy is reached, AutoQuant will return immediately without applying furhter PTQ techniques. Please refer AutoQuant User Guide and API documentation for complete details.


model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
print(f"- Quantized Accuracy (after optimization):  {optimized_accuracy}")


# ---
# ## Summary
# 
# Hope this notebook was useful for you to understand how to use AIMET AutoQuant feature.
# 
# Few additional resources
# - Refer to the AIMET API docs to know more details of the APIs and parameters
# - Refer to the other example notebooks to understand how to use AIMET CLE and AdaRound features in a standalone fashion.


