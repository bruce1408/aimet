
# # Adaptive Rounding (AdaRound)
# This notebook shows a working code example of how to use AIMET to perform Adaptive Rounding (AdaRound).
# 
# AIMET quantization features typically use the "nearest rounding" technique for achieving quantization.
# When using the "nearest rounding" technique, the weight value is quantized to the nearest integer value.
# 
# AdaRound optimizes a loss function using unlabeled training data to decide whether to quantize a specific weight to the closer integer value or the farther one.
# Using AdaRound quantization, a model is able to achieve an accuracy closer to the FP32 model, while using low bit-width integer quantization.
# 
# #### Overall flow
# This notebook covers the following:
# 1. Instantiate the example evaluation and training pipeline
# 2. Convert an FP32 PyTorch model to ONNX and evaluate the model's baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops inserted) and evaluate this simuation model to get a quantized accuracy score
# 4. Apply AdaRound and evaluate the simulation model to get a post-finetuned quantized accuracy score
# 
# #### What this notebook is not
# * This notebook is not designed to show state-of-the-art results
# * For example, it uses a relatively quantization-friendly model like Resnet18
# * Also, some optimization parameters are deliberately chosen to have the notebook execute more quickly


# ---
# ## Dataset
# 
# This notebook relies on the ImageNet dataset for the task of image classification.
# If you already have a version of the dataset readily available, use that.
# Otherwise, download the dataset from an appropriate location (e.g. https://image-net.org/challenges/LSVRC/2012/index.php#).
# 
# **Note1**: The dataloader provided in this example notebook relies on the ImageNet dataset having the following characteristics:
# - Subfolders 'train' for the training samples and 'val' for the validation samples.
# Please see the [pytorch dataset description](https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html) for more details.
# - A subdirectory per class, and a file per each image sample.
# 
# **Note2**: To speed up the execution of this notebook, you may use a reduced subset of the ImageNet dataset.
# E.g. the entire ILSVRC2012 dataset has 1000 classes, 1000 training samples per class and 50 validation samples per class.
# But for the purpose of running this notebook, you could reduce the dataset to 2 samples per class.
# This exercise is left up to the reader and is not necessary.
# 
# Edit the cell below and specify the directory where the downloaded ImageNet dataset is saved.

# DATASET_DIR = '/path/to/dataset/'         # Please replace this with a real directory
DATASET_DIR = '/mnt/share_disk/bruce_trie/outputs/imagenet_dataset'         


# ---
# ## 1. Example evaluation and training pipeline
# 
# The following is an example training and validation loop for this image classification task.
# 
# - **Does AIMET have any limitations on how the training, validation pipeline is written?** Not really. We will see later that AIMET will modify the user's model to create a QuantizationSim model which is still a ONNX model. This QuantizationSim model can be used in place of the original model when doing inference or training.
# - **Does AIMET put any limitation on the interface of the evaluate() or train() methods?** Not really. You should be able to use your existing evaluate and train routines as-is.
# 

import torch, os
import onnxruntime as ort
from Examples.common import image_net_config
from Examples.onnx.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="7"


class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
                                         image_size=image_net_config.dataset['image_size'],
                                         batch_size=image_net_config.evaluation['batch_size'],
                                         is_training=False,
                                         num_workers=image_net_config.evaluation['num_workers']).data_loader
        return data_loader

    @staticmethod
    def evaluate(sess: ort.InferenceSession) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param sess: the model to evaluate
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(sess, iterations=None)


# ---
# ## 2. Convert an FP32 PyTorch model to ONNX and evaluate the model's baseline FP32 accuracy


# For this example notebook, we are going to load a pretrained resnet18 model from torchvision. Similarly, you can load any pretrained PyTorch model instead or convert a model trained in a different framework altogether.

from torchvision.models import resnet18
import onnx

input_shape = (1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
dummy_input = torch.randn(input_shape)
# filename = "./resnet18.onnx"
filename = "/mnt/share_disk/bruce_trie/workspace/logs_aimet/resnet18_origin.onnx"

# Load a pretrained ResNet-18 model in torch
pt_model = resnet18(pretrained=True)

# Export the torch model to onnx
torch.onnx.export(pt_model.eval(),
                  dummy_input,
                  filename,
                  export_params=True,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'},
                  }
                )

model = onnx.load_model(filename)


# ---
# We should decide whether to run the model on a CPU or CUDA device. This example code will use CUDA if available in your onnxruntime environment. You can change this logic and force a device placement if needed.

# cudnn_conv_algo_search is fixing it to default to avoid changing in accuracies/outputs at every inference
if 'CUDAExecutionProvider' in ort.get_available_providers():
    providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
    use_cuda = True
else:
    providers = ['CPUExecutionProvider']
    use_cuda = False


# ---
# Let's create an onnxruntime session and determine the FP32 (floating point 32-bit) accuracy of this model using the evaluate() routine

sess = ort.InferenceSession(filename, providers=providers)
accuracy = ImageNetDataPipeline.evaluate(sess)
print(accuracy)


# ---
# ## 3. Create a quantization simulation model and determine quantized accuracy


# 
# ## Fold Batch Normalization layers
# Before we determine the simulated quantized accuracy using QuantizationSimModel, we will fold the BatchNormalization (BN) layers in the model.
# These layers get folded into adjacent Convolutional layers. The BN layers that cannot be folded are left as they are.
# 
# **Why do we need to this?**
# 
# On quantized runtimes (like TFLite, SnapDragon Neural Processing SDK, etc.), it is a common practice to fold the BN layers.
# Doing so results in an inferences/sec speedup since unnecessary computation is avoided.
# 
# From a floating point compute perspective, a BN-folded model is mathematically equivalent to a model with BN layers from an inference perspective, and produces the same accuracy.
# However, folding the BN layers can increase the range of the tensor values for the weight parameters of the adjacent layers.
# 
# This can have a negative impact on the quantized accuracy of the model (especially when using INT8 or lower precision).
# We want to simulate that on-target behavior by doing BN folding here.
# 
# The following code calls AIMET to fold the BN layers in-place on the given model:

from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight

_ = fold_all_batch_norms_to_weight(model)


# ## Create Quantization Sim Model
# 
# Now we use AIMET to create a QuantizationSimModel. This basically means that AIMET will insert fake quantization ops in the model graph and will configure them.
# A few of the parameters are explained here
# - **quant_scheme**: We set this to "QuantScheme.post_training_tf_enhanced"
#     - Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
# - **default_activation_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all activation quantizations in the model using integer 8-bit precision
# - **default_param_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all parameter quantizations in the model using integer 8-bit precision
# 
# In case the ONNX model has **custom ops**, we need to specify the paths of compiled custom ops via **user_onnx_libs** parameter.
# For example, user_onnx_libs=['path/to/custom_op1.so', 'path/to/custom_op2.so']
# 
# There are other parameters that are set to default values in this example. Please check the AIMET API documentation of QuantizationSimModel to see reference documentation for all the parameters.

import copy
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

sim = QuantizationSimModel(model=copy.deepcopy(model),
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)


# ## Compute Encodings
# Even though AIMET has added 'quantizer' nodes to the model graph, the model is not ready to be used yet.
# Before we can use the sim model for inference or training, we need to find appropriate scale/offset quantization parameters for each 'quantizer' node.
# 
# For activation quantization nodes, we need to pass unlabeled data samples through the model to collect range statistics which will then let AIMET calculate appropriate scale/offset quantization parameters.
# This process is sometimes referred to as calibration. AIMET simply refers to it as 'computing encodings'.
# 
# We create a routine to pass unlabeled data samples through the model.
# This should be fairly simple - use the existing train or validation data loader to extract some samples and pass them to the model.
# We don't need to compute any loss metrics, so we can just ignore the model output. A few pointers regarding the data samples:
# 
# - In practice, we need a very small percentage of the overall data samples for computing encodings.
#   For example, the training dataset for ImageNet has 1M samples. For computing encodings we only need 500 to 1000 samples.
# - It may be beneficial if the samples used for computing encoding are well distributed.
#   It's not necessary that all classes need to be covered since we are only looking at the range of values at every layer activation.
#   However, we definitely want to avoid an extreme scenario like all 'dark' or 'light' samples are used - e.g. only using pictures captured at night might not give ideal results.
# 
# The following shows an example of a routine that passes unlabeled samples through the model for computing encodings.
# This routine can be written in many different ways, this is just an example.

def pass_calibration_data(session, samples):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size
    input_name = sess.get_inputs()[0].name

    batch_cntr = 0
    for input_data, target_data in data_loader:

        inputs_batch = input_data.numpy()
        session.run(None, {input_name : inputs_batch})

        batch_cntr += 1
        if (batch_cntr * batch_size) > samples:
            break


# ---
# Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization encodings.
# Encodings here refer to scale/offset quantization parameters.

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)


# ---
# Now the QuantizationSim model is ready to be used for inference.
# First we can pass this model to the same evaluation routine we used before.
# The evaluation routine will now give us a simulated quantized accuracy score for INT8 quantization instead of the FP32 accuracy score we saw before.

accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# ## 4. Apply Adaround
# 
# We can now apply AdaRound to this model.
# 
# Some of the parameters for AdaRound are described below
# 
# - **dataloader:**  AdaRound needs a dataloader that iterates over unlabeled data for the layer-by-layer optimization to learn the rounding vectors. We should comply with the class signature for the dataloader which is expected by AdaRound.
# - **num_batches:** The number of batches used to evaluate the model while calculating the quantization encodings. Typically we want AdaRound to use around 2000 samples. So with a batch size of 32, this may translate to 64 batches. To speed up the execution here we are using a batch size of 1.
# - **default_num_iterations:** The number of iterations to adaround each layer. Default value is set to 10000 and we strongly recommend to not reduce this number. But in this example we are using 32 to speed up the execution runtime.

import os
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters

# Dataloader satisfying the class signature required by AdaRound
class DataLoader:
    """
    This dataloader derives unlabeled samples in the form of numpy arrays from a torch dataloader
    """
    def __init__(self):
        self._torch_data_loader = ImageNetDataPipeline.get_val_dataloader()
        self._iterator = None
        self.batch_size = self._torch_data_loader.batch_size

    def __iter__(self):
        self._iterator = iter(self._torch_data_loader)
        return self

    def __next__(self):
        input_data, _ = next(self._iterator)
        return input_data.numpy()

    def __len__(self):
        return len(self._torch_data_loader)

data_loader = DataLoader()
params = AdaroundParameters(data_loader=data_loader, num_batches=1, default_num_iterations=32, 
                            forward_fn=pass_calibration_data, forward_pass_callback_args=1000)

output_path = '/mnt/share_disk/bruce_trie/workspace/logs_aimet/adaround_output/' 
os.makedirs(output_path, exist_ok=True)

ada_model = Adaround.apply_adaround(model, params,
                                    path=output_path, 
                                    filename_prefix='adaround', 
                                    default_param_bw=8,
                                    default_quant_scheme=QuantScheme.post_training_tf_enhanced)


# ---
# Now, we can determine the simulated quantized accuracy of the model after applying Adaround.
# We again create a simulation model like before and evaluate to determine simulated quantized accuracy.
# 
# **Note:** There are two important things to understand in the following cell.
#   - **Parameter Biwidth Precision**: The QuantizationSimModel must be created with the same parameter bitwidth precision that was used in the apply_adaround() created.
#     
#   - **Freezing the parameter encodings**:
# After creating the QuantizationSimModel, the set_and_freeze_param_encodings() API must be called before calling the compute_encodings() API.
# While applying AdaRound, the parameter values have been rounded up or down based on these initial encodings internally created.
# For Quantization Simulation accuracy, it is important to freeze these encodings.
# If the parameters encodings are NOT frozen, the call to compute_encodings() will alter the value of the parameters encodings and Quantization Simulation accuracy will not reflect the AdaRounded accuracy.

sim = QuantizationSimModel(model=ada_model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)

sim.set_and_freeze_param_encodings(encoding_path=os.path.join("output", 'adaround.encodings'))

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)


# ---
# Now the QuantizationSim model is ready to be used for inference.
# First we can pass this model to the same evaluation routine we used before.
# The evaluation routine will now give us a simulated quantized accuracy score for INT8 quantization, using the newly AdaRounded model with updated parameters.

accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# Depending on your settings you may have observed a slight gain in accuracy after applying AdaRound.
# The settings used in this notebook are designed only to serve as code examples, designed to run quickly, but may not be optimal.
# Please try this workflow against the model of your choice and play with the number of samples and other parameters to get the best results.
# 
# The next step would be to take this model to target.
# We need to do two things:
# - export the model with the updated weights without the fake quantization ops
# - export the encodings (scale/offset quantization parameters).
# AIMET QuantizationSimModel provides an export API for this purpose.

# sim.export(path='./output/', filename_prefix='resnet18_after_adaround')
sim.export(path=output_path, filename_prefix='resnet18_after_adaround')


# ---
# ## Summary


# This example illustrated how the AIMET AdaRound API is invoked to achieve post training quantization.
# To use AIMET AdaRound for your specific needs, replace the model with your model and replace the data pipeline with your data pipeline.
# As indicated above, some parameters in this example have been chosen in such a way to make this example execute faster.
# 
# We hope this notebook was useful for you to understand how to use AIMET for performing AdaRound.
# 
# A few additional resources:
# - Refer to the AIMET API docs to know more details of the APIs and optional parameters
# - Refer to the other example notebooks to understand how to use AIMET post-training quantization techniques and QAT techniques


