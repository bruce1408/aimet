
# # Quantization simulation
# 
# This notebook contains a working example of AIMET Quantization simulation. 
# QAT is an AIMET feature that adds quantization simulation operations (also called fake quantization ops) to a trained ML model. 
# A standard training pipeline is then used to train or fine-tune the model. 
# The resulting model should show improved accuracy on quantized ML accelerators.
# 
# The quantization parameters (like encoding min/max, scale, and offset) for activations are computed once. During fine-tuning, the model weights are updated to minimize the effects of quantization in the forward pass, keeping the quantization parameters constant.
# 
# ## Overall flow
# 
# The example follows these high-level steps:
# 
# 1. Instantiate the example evaluation pipeline
# 2. Convert an FP32 PyTorch model to ONNX and evaluate the model's baseline FP32 accuracy
# 3. Create a quantization simulation model (with fake quantization ops inserted) and evaluate this simulation model to get a quantized accuracy score.
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


DATASET_DIR = '/path/to/dataset/'         # Please replace this with a real directory


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
# ## 2. Convert an FP32 PyTorch model to ONNX, simplify & then evaluate baseline FP32 accuracy


# **2.1 Load a pretrained resnet18 model from torchvision.** 
# 
# You can load any pretrained PyTorch model instead.


from torchvision.models import resnet18
import onnx

input_shape = (1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
dummy_input = torch.randn(input_shape)
filename = "./resnet18.onnx"

# Load a pretrained ResNet-18 model in torch
pt_model = resnet18(pretrained=True)

# Export the torch model to onnx
torch.onnx.export(pt_model.eval(),
                  dummy_input,
                  filename,
                  training=torch.onnx.TrainingMode.EVAL,
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
# **3.1 Use the following code to call AIMET to fold the BN layers in-place on the model.**


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


from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

sim = QuantizationSimModel(model=model,
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
# **3.4 Call AIMET to pass data through the model and compute the quantization encodings.** 
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


# ## For more information
# 
# See the [AIMET API docs](https://quic.github.io/aimet-pages/AimetDocs/api_docs/index.html) for details about the AIMET APIs and optional parameters.
# 
# See the [other example notebooks](https://github.com/quic/aimet/tree/develop/Examples/torch/quantization) to learn how to use QAT with range-learning and other AIMET post-training quantization techniques.


