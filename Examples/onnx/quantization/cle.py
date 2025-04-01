
# # Cross-Layer Equalization (CLE)
# 
# This notebook showcases a working code example of how to use AIMET to apply Cross-Layer Equalization (CLE). CLE is a post-training quantization techniques that aim to improve quantized accuracy of a given model. CLE does not need any data samples. This technique help recover quantized accuracy when the model quantization is sensitive to parameter quantization as opposed to activation quantization.
# 
# To learn more about this technique, please refer to the "Data-Free Quantization Through Weight Equalization and Bias Correction" paper from ICCV 2019 - https://arxiv.org/abs/1906.04721
# 
# **Cross-Layer Equalization**
# AIMET performs the following steps when running CLE:
# 1. Batch Norm Folding: Folds BN layers into Conv layers immediate before or after the Conv layers.
# 2. Cross-Layer Scaling: Given a set of consecutive Conv layers, equalizes the range of tensor values per-channel by scaling up/down per-channel weight tensor values of a layer and corresponding scaling down/up per-channel weight tensor values of the subsequent layer.
# 3. High Bias Folding: Cross-layer scaling may result in high bias parameter values for some layers. This technique folds some of the bias of a layer into the subsequent layer's parameters.
# 
# 
# #### Overall flow
# This notebook covers the following
# 1. Instantiate the example evaluation and training pipeline
# 2. Load the FP32 model and evaluate the model to find the baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops inserted) and evaluate this simuation model to get a quantized accuracy score
# 4. Apply CLE and evaluate the simulation model to get a post-finetuned quantized accuracy score
# 
# 
# #### What this notebook is not
# * This notebook is not designed to show state-of-the-art results. For example, it uses a relatively quantization-friendly model like Resnet18. Also, some optimization parameters are deliberately chosen to have the notebook execute more quickly.
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


# DATASET_DIR = '/path/to/dataset/'         # Please replace this with a real directory
DATASET_DIR = '/mnt/share_disk/bruce_trie/outputs/imagenet_dataset'         # Replace this path with a real directory


# ---
# 
# ## 1. Example evaluation and training pipeline
# 
# The following is an example training and validation loop for this image classification task.
# 
# - **Does AIMET have any l
# imitations on how the training, validation pipeline is written?** Not really. We will see later that AIMET will modify the user's model to create a QuantizationSim model which is still a PyTorch model. This QuantizationSim model can be used in place of the original model when doing inference or training.
# - **Does AIMET put any limitation on the interface of the evaluate() or train() methods?** Not really. You should be able to use your existing evaluate and train routines as-is.
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
# 
# ## 2. Convert an FP32 PyTorch model to ONNX and evaluate the model's baseline FP32 accuracy


# For this example notebook, we are going to load a pretrained resnet18 model from torchvision. Similarly, you can load any pretrained PyTorch model instead or convert a model trained in a different framework altogether.


from torchvision.models import resnet18
import onnx

input_shape = (1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
dummy_input = torch.randn(input_shape)
# filename = "./resnet18.onnx"
filename = "/mnt/share_disk/bruce_trie/workspace/logs_aimet/resnet18_offical.onnx"

# Load a pretrained ResNet-18 model in torch
pt_model = resnet18(pretrained=True)

# Export the torch model to onnx
torch.onnx.export(pt_model.eval(),
                  dummy_input,
                  filename,
                  export_params=True,
                  do_constant_folding=False,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'},
                  }
                  )

model = onnx.load_model(filename)


# ---
# We should decide whether to place the model on a CPU or CUDA device. This example code will use CUDA if available in your current execution environment. You can change this logic and force a device placement if needed.


# cudnn_conv_algo_search is fixing it to default to avoid changing in accuracies/outputs at every inference
if 'CUDAExecutionProvider' in ort.get_available_providers():
    providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
    use_cuda = True
else:
    providers = ['CPUExecutionProvider']
    use_cuda = False


# ---
# Let's create an onnxruntime session and determine the FP32 (floating point 32-bit) accuracy of this model using the evaluate() routine
# 


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
# On quantized runtimes (like TFLite, SnapDragon Neural Processing SDK, etc.), it is a common practice to fold the BN layers. Doing so, results in an inferences/sec speedup since unnecessary computation is avoided. Now from a floating point compute perspective, a BN-folded model is mathematically equivalent to a model with BN layers from an inference perspective, and produces the same accuracy. However, folding the BN layers can increase the range of the tensor values for the weight parameters of the adjacent layers. And this can have a negative impact on the quantized accuracy of the model (especially when using INT8 or lower precision). So, we want to simulate that on-target behavior by doing BN folding here.
# 
# The following code calls AIMET to fold the BN layers in-place on the given model


from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight

_ = fold_all_batch_norms_to_weight(model)


# ---
# ## Create Quantization Sim Model
# Now we use AIMET to create a QuantizationSimModel. This basically means that AIMET will insert fake quantization ops in the model graph and will configure them.
# A few of the parameters are explained here
# - **quant_scheme**: We set this to "QuantScheme.post_training_tf_enhanced"
#     - Supported options are 'tf_enhanced' or 'tf' or using Quant Scheme Enum QuantScheme.post_training_tf or QuantScheme.post_training_tf_enhanced
# - **default_output_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all activation quantizations in the model using integer 8-bit precision
# - **default_param_bw**: Setting this to 8, essentially means that we are asking AIMET to perform all parameter quantizations in the model using integer 8-bit precision
# - **num_batches**: The number of batches used to evaluate the model while calculating the quantization encodings.Number of batches to use for computing encodings. Only 5 batches are used here to speed up the process. In addition, the number of images in these 5 batches should be sufficient for compute encodings
# - **rounding_mode**: The rounding mode used for quantization. There are two possible choices here - 'nearest' or 'stochastic' We will use "nearest."
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
# Even though AIMET has added 'quantizer' nodes to the model graph but the model is not ready to be used yet. Before we can use the sim model for inference or training, we need to find appropriate scale/offset quantization parameters for each 'quantizer' node. For activation quantization nodes, we need to pass unlabeled data samples through the model to collect range statistics which will then let AIMET calculate appropriate scale/offset quantization parameters. This process is sometimes referred to as calibration. AIMET simply refers to it as 'computing encodings'.
# 
# So we create a routine to pass unlabeled data samples through the model. This should be fairly simple - use the existing train or validation data loader to extract some samples and pass them to the model. We don't need to compute any loss metric etc. So we can just ignore the model output for this purpose. A few pointers regarding the data samples
# 
# In practice, we need a very small percentage of the overall data samples for computing encodings. For example, the training dataset for ImageNet has 1M samples. For computing encodings we only need 500 or 1000 samples.
# It may be beneficial if the samples used for computing encoding are well distributed. It's not necessary that all classes need to be covered etc. since we are only looking at the range of values at every layer activation. However, we definitely want to avoid an extreme scenario like all 'dark' or 'light' samples are used - e.g. only using pictures captured at night might not give ideal results.
# The following shows an example of a routine that passes unlabeled samples through the model for computing encodings. This routine can be written in many different ways, this is just an example.


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
# Now the QuantizationSim model is ready to be used for inference or training. First we can pass this model to the same evaluation routine we used before. The evaluation routine will now give us a simulated quantized accuracy score for INT8 quantization instead of the FP32 accuracy score we saw before.


accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# ## 4. 1 Cross Layer Equalization
# 
# The next cell performs cross-layer equalization on the model. As noted before, the function folds batch norms, applies cross-layer scaling, and then folds high biases.
# 
# **Note:** Interestingly, CLE needs BN statistics for its procedure. If a BN folded model is provided, CLE will run the CLS (cross-layer scaling) optimization step but will skip the HBA (high-bias absorption) step. To avoid this, we simply load the original model again before running CLE.
# 
# **Note:** CLE equalizes the model in-place


# filename = "./resnet18.onnx"
filename = "/mnt/share_disk/bruce_trie/workspace/logs_aimet/resnet18_offical.onnx"
model = onnx.load_model(filename)


from aimet_onnx.cross_layer_equalization import equalize_model
equalize_model(model)


# ---
# Now, we can determine the simulated quantized accuracy of the equalized model. We again create a simulation model like before and evaluate to determine simulated quantized accuracy.


sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_activation_bw=8,
                           default_param_bw=8,
                           use_cuda=use_cuda)

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=1000)

accuracy = ImageNetDataPipeline.evaluate(sim.session)
print(accuracy)


# ---
# ## Summary
# 
# Hope this notebook was useful for you to understand how to use AIMET for performing Cross Layer Equalization (CLE).
# 
# Few additional resources
# - Refer to the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) to know more details of the APIs and optional parameters
# - Refer to the other example notebooks to understand how to use AIMET post-training quantization techniques and QAT techniques


