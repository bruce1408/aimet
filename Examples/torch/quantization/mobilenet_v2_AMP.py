

# # Automatic Mixed-Precision (AMP)
# 
# This notebook shows a working code example of how to use AIMET to perform Auto Mixed Precision (AMP). AMP is a technique where given a quantized accuracy target, AIMET finds bit-precision per-layer to meet that accuracy target while trying to optimize the model for inference speed.
# 
# As an example, say a particular model is not meeting a desired accuracy target when run in INT8. The Auto Mixed Precision feature will find a minimal set of layers that need to run on say INT16 to get to the desired accuracy. It should be noted that choosing higher precision for some layers necessarily involves a trade-off: lower inferences/sec for higher accuracy and vice-versa.
# 
# Alternatively, the AMP feature can be used to generate a pareto curve (accuracy vs. bit-ops) that can guide the user to decide the right operating point for this tradeoff.
# 
# This notebook specifically shows working code example for the above. 
# 
# #### Overall flow
# This notebook covers the following
# 1. Instantiate the example evaluation pipeline
# 2. Load the FP32 model and evaluate the model to find the baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops inserted)
# 4. Run AMP algorithm on the quantized model 
# 
# #### What this notebook is not
# * This notebook is not designed to show state-of-the-art AMP results. For example, it uses a relatively quantization-friendly model like Resnet18. Also, some optimization parameters like number of samples for evaluation are deliberately chosen to have the notebook execute more quickly.


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


# DATASET_DIR = '/path/to/dataset'         # Please replace this with a real directory


# ---
# ## 1. Example evaluation pipeline
# 
# The following is an example validation loop for this image classification task.


import os
import torch
from torchvision.datasets.imagenet import ImageFolder
from torchvision import transforms as T
from tqdm import tqdm
from Examples.common import image_net_config
from Examples.common import config_param
 
DATASET_DIR = config_param.imagenet_dir
log_dir = f"{config_param.aimet_log_dir}/mobilenet_v2_AMP"

def get_val_dataloader(num_samples=None):
    root = os.path.join(DATASET_DIR, 'val')
    image_size = image_net_config.dataset['image_size']
    
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=image_net_config.dataset['images_mean'],
                    std=image_net_config.dataset['images_std']),
    ])
    dataset = ImageFolder(root, transforms)

    if num_samples is not None:
        assert num_samples <= len(dataset)
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(num_samples, generator=generator).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    batch_size = image_net_config.evaluation['batch_size']
    num_workers = image_net_config.evaluation['num_workers']
    
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        pin_memory=True)


def evaluate_accuracy(model, num_samples=None):
    data_loader = get_val_dataloader(num_samples)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()

    top1_accuracy = 0.0
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for input_data, target_data in tqdm(data_loader):
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            logit = model(input_data)
            pred = softmax(logit).topk(1).indices.squeeze()

            top1_accuracy += (pred == target_data).sum()
    
    return float(top1_accuracy / len(data_loader.dataset))


# ---
# ## 2. Load the model and evaluate to get a baseline FP32 accuracy score


# For this example notebook, we are going to load a pretrained resnet18 model from torchvision. Similarly, you can load any pretrained PyTorch model instead.


from torchvision.models import resnet18

model = resnet18(pretrained=True)


# AIMET quantization simulation requires the user's model definition to follow certain guidelines. For example, functionals defined in forward pass should be changed to equivalent torch.nn.Module.
# AIMET user guide lists all these guidelines.
# The following **ModelPreparer** API uses new graph transformation feature available in PyTorch 1.9+ version and automates model definition changes required to comply with the above guidelines. 


from aimet_torch.model_preparer import prepare_model

model = prepare_model(model)


# ---
# We should decide whether to place the model on a CPU or CUDA device. This example code will use CUDA if available in your current execution environment. You can change this logic and force a device placement if needed.


use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))


# ---
# Let's determine the FP32 (floating point 32-bit) accuracy of this model using the evaluate_accuracy() routine


accuracy = evaluate_accuracy(model)
print(accuracy)


# ---
# ## 3. Create a quantization simulation model
# 
# ## Fold Batch Normalization layers
# Before we determine the simulated quantized accuracy using QuantizationSimModel, we will fold the BatchNormalization (BN) layers in the model. These layers get folded into adjacent Convolutional layers. The BN layers that cannot be folded are left as they are.
# 
# **Why do we need to this?**
# On quantized runtimes (like TFLite, SnapDragon Neural Processing SDK, etc.), it is a common practice to fold the BN layers. Doing so, results in an inferences/sec speedup since unnecessary computation is avoided. Now from a floating point compute perspective, a BN-folded model is mathematically equivalent to a model with BN layers from an inference perspective, and produces the same accuracy. However, folding the BN layers can increase the range of the tensor values for the weight parameters of the adjacent layers. And this can have a negative impact on the quantized accuracy of the model (especially when using INT8 or lower precision). So, we want to simulate that on-target behavior by doing BN folding here.
# 
# The following code calls AIMET to fold the BN layers in-place on the given model


from aimet_torch.batch_norm_fold import fold_all_batch_norms

_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))


# ## Create Quantization Sim Model
# 
# Now we use AIMET to create a QuantizationSimModel. This basically means that AIMET will insert fake quantization ops in the model graph and will configure them.
# 
# A few of the parameters are explained here
# - **quant_scheme**: We set this to "QuantScheme.post_training_tf_enhanced"
#     - Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
# - **default_output_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all activation quantizations in the model using integer 8-bit precision
# - **default_param_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all parameter quantizations in the model using integer 8-bit precision
# 
# There are other parameters that are set to default values in this example. Please check the AIMET API documentation of QuantizationSimModel to see reference documentation for all the parameters.


from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()

sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)


# ---
# We can check the modifications AIMET has made to the model graph. One way is to print the model, and we can see that AIMET has added quantization wrapper layers. Note: use sim.model to access the modified PyTorch model. By default, AIMET creates a copy of the original model prior to modifying it. There is a parameter to override this behavior.


print(sim.model)


# ---
# We can also check how AIMET has configured the added fake quantization nodes, which AIMET refers to as 'quantizers'. You can see this by printing the sim object.


print(sim)


# ---
# ## Compute Encodings
# 
# Even though AIMET has added 'quantizer' nodes to the model graph but the model is not ready to be used yet. Before we can use the sim model for inference or training, we need to find appropriate scale/offset quantization parameters for each 'quantizer' node. For activation quantization nodes, we need to pass unlabeled data samples through the model to collect range statistics which will then let AIMET calculate appropriate scale/offset quantization parameters. This process is sometimes referred to as calibration. AIMET simply refers to it as 'computing encodings'.
# 
# It may be beneficial if the samples used for forward pass are well distributed, though it doesn't necessarily mean that all classes need to be covered etc. since we are only looking at the range of values at every layer activation. However, we definitely want to avoid using extremely biased subset of the original dataset, such as a subset consisting of only 'dark' or 'light' images.
# 
# The following shows an example of a routine that passes unlabeled samples through the model for computing encodings. This routine can be written in many different ways, this is just an example.


def pass_calibration_data(model, num_samples):
    data_loader = get_val_dataloader(num_samples)

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()

    with torch.no_grad():
        for input_data, _ in tqdm(data_loader):
            inputs_batch = input_data.to(device)
            model(inputs_batch)


# ---
# Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization encodings. Encodings here refer to scale/offset quantization parameters.


sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000) # Use 10000 samples for computing initial scale/offset

accuracy = evaluate_accuracy(sim.model)
print(accuracy)


# ## 4. Run AMP algorithm  on the quantized model


# AMP algorithm runs in 3 phases (phase-3 is optional). Phase-1 comprises of calculating the sensitivity for each layer. Phase-2 comprises of greedily selecting which layers should have what bitwidth based on options provided by the user. Phase-3 derives a set of mixed-precision solutions having less bitwidth convert op overhead compared to original phase-2 solution. For phase 1 and phase 2 we require to pass data through the model.


# ### Define callback functions for AMP
# 
# AMP requires three callback functions, `forward_pass_callback`, `eval_callback_for_phase1`, and `eval_callback_for_phase2`.
# 
# `forward_pass_callback` is a CallbackFunc object which is required for computing initial scale/offset as explained above. In this example, we will reuse the same forward function as the previous code snippet.


from aimet_common.defs import CallbackFunc

# Use 1000 samples for computing initial scale/offset
forward_pass_callback = CallbackFunc(pass_calibration_data, func_callback_args=1000)


# `eval_callback_for_phase1` and `eval_callback_for_phase2` are also CallbackFunc objects used for measuring the model's eval score in phase 1 and phase 2, respecitively. Even though they are both used for evaluating model's quality, they have slightly different goals. Eval callback for phase 1 is used to get a rough measure of the model's quality, whereas eval callbak for phase 2 is used for measuring the model's quality in the real practice. This implies that the eval callback for phase 1 can be more flexible than phase 2.
# 
# For example, to measure your model's quality in phase 1, you can use relatively smaller dataset, or even use an indirect measure (e.g. SQNR between the fp32 outputs and fake-quantized outputs) that can be computed faster than but correlates well with the real metric. 


from aimet_torch.amp.mixed_precision_algo import EvalCallbackFactory

# Phase 1 evaluation: Evaluate SQNR between fp32 outputs and fake-quantized outputs
def forward_one_batch(model, batch):
    image, label = batch
    
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return model(image.to(device))

eval_callback_factory =  EvalCallbackFactory(get_val_dataloader(300),
                                             forward_fn=forward_one_batch)
eval_callback_for_phase1 = eval_callback_factory.sqnr()

# Alternatively, you can also evaluate the classification accuracy with a small subset of validation dataset
###
# eval_callback_for_phase1 = CallbackFunc(evaluate_accuracy, func_callback_args=1000) # Use 1000 samples for phase 1 evaluation
###


# In phase 2, on the other hand, the eval callback should desirably measure the model's quality in the real metric (e.g. accuracy, mIoU, etc.) with full validation dataset.


# Phase 2 evaluation: Evaluate accuracy with full dataset
eval_callback_for_phase2 = CallbackFunc(evaluate_accuracy, func_callback_args=None) # Use full dataset for phase 2 evaluation


# ## Parameters for AMP algorithm
# 
# A few of the parameters required for AMP are explained below
# 
# - **sim**: `QuantizationSimModel` object to which mixed precision will be applied.
# - **dummy_inputs**: Dummy input to the model. If the model has more than one input, pass a tuple. User is expected to place the tensors on the appropriate device.
# - **eval_callback_for_phase1**: A CallbackFunc object used for measure sensitivity of each quantizer group in phase 1. The phase 1 involves finding accuracy list/sensitivity of each module. Therefore, a user might want to run the phase 1 with a smaller dataset
# - **eval_callback_for_phase2**: A CallbackFunc object used for evaluating accuracy of quantized model in phase 2. The phase 2 involves finding pareto front curve.
# - **candidates**: It is a list of tuples for all possible bitwidth values for activations and parameters. Suppose the possible combinations are-((Activation bitwidth - 8, Activation data type - int), (Parameter bitwidth - 16, parameter data type - int)) ((Activation bitwidth - 16, Activation data type - float), (Parameter bitwidth - 16, parameter data type - float)) candidates will be [((8, QuantizationDataType.int), (16, QuantizationDataType.int)), ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
# - **allowed_accuracy_drop**: Maximum allowed drop in accuracy from FP32 baseline. The pareto front curve is plotted only till the point where the allowable accuracy drop is met. To get a complete plot for picking points on the curve, the user can set the allowable accuracy drop to None.
# - **results_dir**: Path to save results and cache intermediate results
# - **clean_start**: If true, any cached information from previous runs will be deleted prior to starting the mixed-precision analysis. If false, prior cached information will be used if applicable. Note it is the user's responsibility to set this flag to true if anything in the model or quantization parameters changes compared to the previous run.
# - **use_all_amp_candidates**: Using the “supported_kernels” field in the config file (under defaults and op_type sections), a list of supported candidates can be specified. All the AMP candidates which are passed through the “candidates” field may not be supported based on the data passed through “supported_kernels”. When the field “use_all_amp_candidates” is set to True, the AMP algorithm will ignore the "supported_kernels" in the config file and continue to use all candidates.
# - **forward_pass_callback** A CallbackFunc object which used as forward pass callback for computing quantization encodings.
# - **amp_search_algo**: An AMPSearchAlgo enum that defines the search algorithm to be used for phase 2. You can choose one of `AMPSearchAlgo.Binary` (default), `AMPSearchAlgo.Interpolation`, and `AMPSearchAlgo.BruteForce`.
# - **phase1_optimize**: It is a flag for phase 1 implementation which used to choose either optmized or default implementation. If user set this parameter to false then phase1 default logic will be executed else optimized logic will be executed.


from aimet_common.defs import QuantizationDataType
from aimet_common.amp.utils import AMPSearchAlgo

candidates = [
    ((16, QuantizationDataType.int), (16, QuantizationDataType.int)), 
    ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
    ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
    ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
]

allowed_accuracy_drop = 0.001 # Allow 0.1%p accuracy drop

results_dir = '/mnt/share_disk/bruce_trie/workspace/logs_aimet'

amp_search_algo = AMPSearchAlgo.Binary


# # Call AMP API
# 
# AMP algorithms changes the quantization sim model in place and the final result after running the AMP API is a model which meets the accuracy goal. The algorithm also returns a pareto curve which is a plot between the bit ops and accuracy. Bit ops is calculated by multiplying the MAC's required by the layer, it's output bitwidth & it's parameter bitwidth. Therefore, as we lower the bitwidth for a given layer, bit ops reduces, thereby implying that lesser compute is needed for that layer. 
# 
# Looking at the pareto curve, a user can decide if they want to change the accuracy drop. Note: If a user sets clean_start to False and change the allowed accuracy drop, then AMP will use cached data from results directory so that re-computation is avoided. 


from aimet_torch.mixed_precision import choose_mixed_precision
from aimet_torch.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo

# Enable phase-3 (optional)
# GreedyMixedPrecisionAlgo.ENABLE_CONVERT_OP_REDUCTION = True
# Note: supported candidates ((8,int), (8,int)) & ((16,int), (8,int))

pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, 
                                           eval_callback_for_phase1=eval_callback_for_phase1, 
                                           eval_callback_for_phase2=eval_callback_for_phase2, 
                                           allowed_accuracy_drop=allowed_accuracy_drop, 
                                           results_dir=results_dir,
                                           clean_start=True, 
                                           forward_pass_callback=forward_pass_callback,
                                           amp_search_algo=amp_search_algo,
                                           phase1_optimize= True)


# ---
# So we have a Mixed precision model after AMP. Now the next step would be to actually take this model to target. For this purpose, we need to export the model. 


os.makedirs('/mnt/share_disk/bruce_trie/workspace/logs_aimet', exist_ok=True)
dummy_input = dummy_input.cpu()
sim.export(path='/mnt/share_disk/bruce_trie/workspace/logs_aimet', filename_prefix='resnet18_mixed_precision', dummy_input=dummy_input)


# ---
# ## Summary
# 
# Hope this notebook was useful for you to understand how to use AIMET for performing QAT with range-learning.
# 
# Few additional resources
# - Refer to the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) to know more details of the APIs and optional parameters.
# - Refer to the [other example notebooks](https://github.com/quic/aimet/tree/develop/Examples/torch/quantization).


