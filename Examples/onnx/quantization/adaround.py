
# # Adaptive Rounding (AdaRound)
# This notebook contains a working example of AIMET adaptive rounding (AdaRound).
# 
# AIMET quantization features typically use the "nearest rounding" technique for achieving quantization.
# When using the nearest rounding technique, the weight value is quantized to the nearest integer value.
# 
# AdaRound optimizes a loss function using unlabeled training data to decide whether to quantize a specific weight to the closer integer value or the farther one.
# Using AdaRound, quantized accuracy is closer to the FP32 model than with nearest rounding.
# 
# ## Overall flow
# 
# The example follows these high-level steps:
# 
# 1. Instantiate the example evaluation and training pipeline
# 2. Load the FP32 model and evaluate the model to find the baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops) and evaluate the quantized simuation model
# 4. Apply AdaRound and evaluate the simulation model to get a post-finetuned quantized accuracy score
# 
# 
# <div class="alert alert-info">
# 
# Note
# 
# This notebook does not show state-of-the-art results. For example, it uses a relatively quantization-friendly model (Resnet18). Also, some optimization parameters like number of fine-tuning epochs are chosen to improve execution speed in the notebook.
# 
# </div>


# ---
# 
# ## Dataset
# 
# This example does image classification on the ImageNet dataset. If you already have a version of the data set, use that. Otherwise download the data set, for example from https://image-net.org/challenges/LSVRC/2012/index .
# 
# <div class="alert alert-info">
# 
# Note
# 
# The dataloader provided in this example relies on these features of the ImageNet data set:
# 
# - Subfolders `train` for the training samples and `val` for the validation samples. See the [pytorch dataset description](https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html) for more details.
# - One subdirectory per class, and one file per image sample.
# 
# </div>
# 
# <div class="alert alert-info">
# 
# Note
# 
# To speed up the execution of this notebook, you can use a reduced subset of the ImageNet dataset. For example: The entire ILSVRC2012 dataset has 1000 classes, 1000 training samples per class and 50 validation samples per class. However, for the purpose of running this notebook, you can reduce the dataset to, say, two samples per class.
# 
# </div>
# 
# Edit the cell below to specify the directory where the downloaded ImageNet dataset is saved.


# Replace this path with a real directory
DATASET_DIR = '/mnt/share_disk/bruce_trie/outputs/imagenet_dataset'         

# import sys
# from spectrautils.onnx_utils import visualize_onnx_model_weights

# adaround_path = "/mnt/share_disk/bruce_trie/workspace/logs_aimet/adaround_output/resnet18_after_adaround.onnx"

# visualize_onnx_model_weights(adaround_path, model_name="resnet18_adaround", results_dir="/mnt/share_disk/bruce_trie/workspace/logs_aimet/adaround_output")


# sys.exit(0)  # 这会完全退出Python解释器
# ---
# 
# ## 1. Instantiate the example training and validation pipeline
# 
# **Use the following training and validation loop for the image classification task.**
# 
# Things to note:
# 
# - AIMET does not put limitations on how the training and validation pipeline is written. AIMET modifies the user's model to create a QuantizationSim model, which is still a PyTorch model. The QuantizationSim model can be used in place of the original model when doing inference or training.
# - AIMET doesn not put limitations on the interface of the `evaluate()` or `train()` methods. You should be able to use your existing evaluate and train routines as-is.
# 


import torch,os
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
# ## 2. Convert an FP32 PyTorch model to ONNX, simplify & then evaluate baseline FP32 accuracy


# **2.1 Load a pretrained resnet18 model from torchvision.** 
# 
# You can load any pretrained PyTorch model instead.


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
# 
# **2.2 It is recommended to simplify the model before using AIMET**


from onnxsim import simplify

try:
    model, _ = simplify(model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')


# ---
# 
# **2.3 Decide whether to place the model on a CPU or CUDA device.** 
# 
# This example uses CUDA if it is available. You can change this logic and force a device placement if needed.


# cudnn_conv_algo_search is fixing it to default to avoid changing in accuracies/outputs at every inference
if 'CUDAExecutionProvider' in ort.get_available_providers():
    providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
    use_cuda = True
else:
    providers = ['CPUExecutionProvider']
    use_cuda = False


# ---
# **2.4 Create an onnxruntime session and determine the FP32 accuracy of this model using the evaluate() routine.**


sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
accuracy = ImageNetDataPipeline.evaluate(sess)
print(accuracy)


# ---
# 
# ## 3. Create a quantization simulation model and determine quantized accuracy
# 
# ### Fold Batch Normalization layers
# 
# Before calculating the simulated quantized accuracy using QuantizationSimModel, fold the BatchNormalization (BN) layers into adjacent Convolutional layers. The BN layers that cannot be folded are left as they are.
# 
# BN folding improves inference performance on quantized runtimes but can degrade accuracy on these platforms. This step simulates this on-target drop in accuracy. 
# 
# **3.1 Use the following code to call AIMET to fold the BN layers in-place on the given model.**


from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight

_ = fold_all_batch_norms_to_weight(model)


# ### Create the Quantization Sim Model
# 
# **3.2 Use AIMET to create a QuantizationSimModel.**
# 
#  In this step, AIMET inserts fake quantization ops in the model graph and configures them.
# 
# Key parameters:
# 
# - Setting **default_output_bw** to 8 performs all activation quantizations in the model using integer 8-bit precision
# - Setting **default_param_bw** to 8 performs all parameter quantizations in the model using integer 8-bit precision
# 
# See [QuantizationSimModel in the AIMET API documentation](https://quic.github.io/aimet-pages/AimetDocs/api_docs/torch_quantsim.html#aimet_torch.quantsim.QuantizationSimModel.compute_encodings) for a full explanation of the parameters.


import copy
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

sim = QuantizationSimModel(model=copy.deepcopy(model),
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)


# ---
# AIMET has added quantizer nodes to the model graph, but before the sim model can be used for inference or training, scale and offset quantization parameters must be calculated for each quantizer node by passing unlabeled data samples through the model to collect range statistics. This process is sometimes referred to as calibration. AIMET refers to it as "computing encodings".
# 
# **3.3 Create a routine to pass unlabeled data samples through the model.** 
# 
# The following code is one way to write a routine that passes unlabeled samples through the model to compute encodings. It uses the existing train or validation data loader to extract samples and pass them to the model. Since there is no need to compute loss metrics, it ignores the model output.  


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


# A few notes regarding the data samples:
# 
# - A very small percentage of the data samples are needed. For example, the training dataset for ImageNet has 1M samples; 500 or 1000 suffice to compute encodings.
# - The samples should be reasonably well distributed. While it's not necessary to cover all classes, avoid extreme scenarios like using only dark or only light samples. That is, using only pictures captured at night, say, could skew the results.
# 
# ---
# 
# **3.4 Call AIMET to use the routine to pass data through the model and compute the quantization encodings.** 
# 
# Encodings here refer to scale and offset quantization parameters.


sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)


# ---
# 
# The QuantizationSim model is now ready to be used for inference or training. 
# 
# **3.5 Pass the model to the same evaluation routine as before to calculate a simulated quantized accuracy score for INT8 quantization for comparison with the FP32 score.**


accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# ## 4. Apply Adaround
# 
# **4.1 Use the code below to apply Adaround to the original model.**
# 
# Some key parameters:
# 
# - **dataloader:**  is a training or validation dataloader. Adaround needs a dataloader in order to use data samples to learn the rounding vectors.
# - **num_batches:** is the number of batches used while calculating the quantization encodings. A typical value for Adaround is 2000 samples. To speed up the execution this example uses a batch size of one.
# - **default_num_iterations:** is the number of iterations to apply to each layer. Default value is 10000, and we strongly recommend using at least this number. This example uses 32 to speed up execution.


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
# **4.2 Quantize the Adarounded model.** 
# 
# <div class="alert alert-info">
# 
# Note
# 
# Two important points about the following code:
# 
# </div>
# 
# - **Parameter Biwidth Precision**: The QuantizationSimModel must be created with the same parameter bitwidth precision that was used in `apply_adaround()`.
#     
# - **Freezing the parameter encodings**:
# After creating the QuantizationSimModel, you must call `set_and_freeze_param_encodings()` before calling `compute_encodings()`.
# During AdaRound, the parameters are rounded based on these initial internally created encodings.
# To maintain accuracy, it is important to freeze these encodings so that the call to `compute_encodings()` does not alter the parameter encodings negate the AdaRounded accuracy.


sim = QuantizationSimModel(model=ada_model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)

sim.set_and_freeze_param_encodings(encoding_path=os.path.join(output_path, 'adaround.encodings'))

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)


# **4.3 Compute the accuracy of the Adarounded model.** 
# 
# Evaluate the simulation model as before to determine simulated quantized accuracy.


accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# There might be little gain in accuracy after this limited application of Adaround. Experiment with the hyper-parameters to get better results.
# 
# ## Next steps
# 
# **Export the model and encodings.**
# 
# - Export the model with the updated weights but without the fake quant ops. 
# - Export the encodings (scale and offset quantization parameters). AIMET QuantizationSimModel provides an export API for this purpose.
# 
# The following code performs these exports.


sim.export(path=output_path, filename_prefix='resnet18_after_adaround')


# ## For more information
# 
# See the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) for details about the AIMET APIs and optional parameters.
# 
# See the [other example notebooks](https://github.com/quic/aimet/tree/develop/Examples/torch/quantization) to learn how to use other AIMET post-training quantization techniques.


