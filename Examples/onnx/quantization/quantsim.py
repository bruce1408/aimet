
# # Quantization Simulation
# 
# This notebook shows a working code example of how to use AIMET to perform quantization simulation (quantsim). Quantsim is an AIMET feature that adds quantization simulation ops (also called fake quantization ops sometimes) to a trained ML model in order to compute quantization encodings and estimate the resulting accuracy of the model when deployed on quantized ML accelerators.
# 
# 
# 
# #### Overall flow
# This notebook covers the following
# 1. Instantiate the example evaluation pipeline
# 2. Convert an FP32 PyTorch model to ONNX and evaluate the model's baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops inserted) and evaluate this simulation model to get a quantized accuracy score
# 
# #### What this notebook is not
# * This notebook is not designed to show state-of-the-art quantized accuracy. For example, it uses a relatively quantization-friendly model like Resnet18. Also, optimization techniques such as Quantization-Aware Training, AdaRound, and Cross-Layer Equalization can be employed to improve the accuracy of quantized models.


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


# DATASET_DIR = '/path/to/dataset/'         # Please replace this with a real directory
DATASET_DIR = '/mnt/share_disk/bruce_trie/outputs/imagenet_dataset'         


# ---
# ## 1. Example evaluation pipeline
# 
# The following is an example validation loop for this image classification task.
# 
# - **Does AIMET have any limitations on how the validation pipeline is written?** Not really. We will see later that AIMET will modify the user's model and provide a QuantizationSim session that acts as a regular onnxruntime inference session. However, it is recommended that users only use inference sessions created by the QuantizationSimModel, as this will automatically register the required custom operators.
# 


import torch
import onnxruntime as ort
from Examples.common import image_net_config
from Examples.onnx.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

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
filename = "/mnt/share_disk/bruce_trie/workspace/logs_aimet/resnet18.onnx"

# Load a pretrained ResNet-18 model in torch
pt_model = resnet18(pretrained=True)

# Export the torch model to onnx
torch.onnx.export(pt_model.eval(),
                  dummy_input,
                  filename,
                  training=torch.onnx.TrainingMode.EVAL,
                  export_params=True,
                  do_constant_folding=True,
                  opset_version=13,                         # 明确指定 opset 版本
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
# Before we determine the simulated quantized accuracy using QuantizationSimModel, we will fold the BatchNormalization (BN) layers in the model. These layers get folded into adjacent Convolutional layers. The BN layers that cannot be folded are left as they are.
# 
# **Why do we need to this?**
# On quantized runtimes (like TFLite, SnapDragon Neural Processing SDK, etc.), it is a common practice to fold the BN layers. Doing so results in an inferences/sec speedup since unnecessary computation is avoided. Now from a floating point compute perspective, a BN-folded model is mathematically equivalent to a model with BN layers from an inference perspective, and produces the same accuracy. However, folding the BN layers can increase the range of the tensor values for the weight parameters of the adjacent layers. And this can have a negative impact on the quantized accuracy of the model (especially when using INT8 or lower precision). So, we want to simulate that on-target behavior by doing BN folding here.
# 
# The following code calls AIMET to fold the BN layers in-place on the given model
# 


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


from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)


# ---
# ## Compute Encodings
# Even though AIMET has added 'quantizer' nodes to the model graph, the model is not ready to be used yet. Before we can use the sim model for inference, we need to find appropriate scale/offset quantization parameters for each 'quantizer' node. For activation quantization node, we need to pass unlabeled data samples through the model to collect range statistics which will then let AIMET calculate appropriate scale/offset quantization parameters. This process is sometimes referred to as calibration. AIMET simply refers to it as 'computing encodings'.
# 
# So we create a routine to pass unlabeled data samples through the model. This should be fairly simple - use the existing train or validation data loader to extract some samples and pass them to the model. We don't need to compute any loss metric etc. So we can just ignore the model output for this purpose. A few pointers regarding the data samples
# - In practice, we need a very small percentage of the overall data samples for computing encodings. For example, the training dataset for ImageNet has 1M samples. For computing encodings we only need 500 or 1000 samples.
# - It may be beneficial if the samples used for computing encoding are well distributed. It's not necessary that all classes need to be covered etc. since we are only looking at the range of values at every layer activation. However, we definitely want to avoid an extreme scenario like all 'dark' or 'light' samples are used - e.g. only using pictures captured at night might not give ideal results.
# 
# The following shows an example of a routine that passes unlabeled samples through the model for computing encodings. This routine can be written in many ways, this is just an example.


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
# Now we call AIMET to use the above routine to pass data through the model and then subsequently compute the quantization encodings. Encodings here refer to scale/offset quantization parameters.


sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)


# ---
# Now the QuantizationSim model is ready to be used for inference. First we can pass this model to the same evaluation routine we used before. The evaluation routine will now give us a simulated quantized accuracy score for INT8 quantization instead of the FP32 accuracy score we saw before.


accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ## Summary
# 
# Hope this notebook was useful for you to understand how to use AIMET for performing QuantizationSimulation.
# 
# Additional resources
# - Refer to the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) to know more details of the APIs and optional parameters.


