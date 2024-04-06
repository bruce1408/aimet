# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import pytest
import torch
import torch.nn.functional as F
from aimet_torch.experimental.v2.quantization.backends import get_backend
from aimet_torch.experimental.v2.quantization.quantizers.affine import Quantize, QuantizeDequantize
from aimet_torch.experimental.v2.nn.true_quant import (
    QuantizedGELU,
    QuantizedLinear,
    QuantizationMixin,
    QuantizedSigmoid,
    QuantizedSoftmax,
    QuantizedLayerNorm,
)
from aimet_torch.experimental.v2.quantization.encodings import AffineEncoding
from aimet_torch.experimental.v2.quantization.quantized_tensor import QuantizedTensor, DequantizedTensor
from aimet_torch.experimental.v2.nn import fake_quant
from aimet_torch.experimental.v2.utils import enable_recompute


def affine_quantize(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    offset: torch.Tensor,
                    bitwidth: int) -> QuantizedTensor:
    """
    Quantizes the input tensor into a QuantizedTensor using the quantization parameters
    """
    tensor_q = get_backend().quantize(tensor, scale, offset, bitwidth)
    encoding = AffineEncoding(scale, offset, bitwidth)
    qtensor = tensor_q.as_subclass(QuantizedTensor)
    qtensor.encoding = encoding
    return qtensor


@pytest.fixture
def input():
    return torch.arange(-5, 5).expand(10, 10) / 10


@pytest.fixture(autouse=True)
def register_int_linear():
    def int_linear(input, weight, bias=None, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        assert isinstance(input, QuantizedTensor)
        assert isinstance(weight, QuantizedTensor)

        input = input.dequantize()
        weight = weight.dequantize()

        return affine_quantize(input.mm(weight.t()) + bias,
                               output_encodings.scale,
                               output_encodings.offset,
                               output_encodings.bitwidth)

    QuantizedLinear.set_default_kernel(int_linear)
    yield
    QuantizedLinear.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_activation():
    def wrap_functional(func):
        def wrapped_func(*args, output_encodings=None, **kwargs):
            # Implicit dequantization is not supported yet
            x, *others = args
            assert isinstance(x, QuantizedTensor)
            output = func(x.dequantize(), *others, **kwargs)
            return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

        return wrapped_func

    QuantizedSoftmax.set_default_kernel(wrap_functional(F.softmax))
    QuantizedSigmoid.set_default_kernel(wrap_functional(torch.sigmoid))
    QuantizedLayerNorm.set_default_kernel(wrap_functional(F.layer_norm))
    QuantizedGELU.set_default_kernel(wrap_functional(F.gelu))
    yield
    QuantizedGELU.set_default_kernel(None)
    QuantizedLayerNorm.set_default_kernel(None)
    QuantizedSigmoid.set_default_kernel(None)
    QuantizedSoftmax.set_default_kernel(None)


@pytest.fixture(autouse=True)
def register_int_layernorm():
    def int_layernorm(input, normalized_shape, weight, bias, eps, *, output_encodings=None):
        # Implicit dequantization is not supported yet
        assert isinstance(input, QuantizedTensor)
        assert isinstance(weight, QuantizedTensor)

        input = input.dequantize()
        weight = weight.dequantize()

        output = F.layer_norm(input.dequantize(), normalized_shape, weight, bias, eps)
        return affine_quantize(output, output_encodings.scale, output_encodings.offset, output_encodings.bitwidth)

    QuantizedLayerNorm.set_default_kernel(int_layernorm)
    yield
    QuantizedLayerNorm.set_default_kernel(None)



class TestTrueQuantLinear:

    def test_no_quantizers(self, input):
        """
        Given: TrueQuantLinear with no input, output, or param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        """
        When: inspect input/output/param quantizers
        Then: quantizers are None
        """
        assert quant_linear.input_quantizers[0] is None
        assert quant_linear.output_quantizers[0] is None
        assert quant_linear.param_quantizers["weight"] is None
        assert quant_linear.param_quantizers["bias"] is None
        """
        When: call forward pass within compute encodings context
        Then: output is equal to floating point output
        """
        expected_output = F.linear(input, quant_linear.weight, quant_linear.bias)
        with quant_linear.compute_encodings():
            output = quant_linear(input)
        assert torch.all(output == expected_output)
        """
        When: call forward pass outside of compute encodings context
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

    def test_fully_specified_quantizers(self, input):
        """
        Given: TrueQuantLinear with input, output, and param quantizers
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.input_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        """
        When: Call forward pass before computing encodings
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass within compute_encodings context
        Then: Output should be equal to fake quant forward pass with activation quantizers disabled
        """
        with quant_linear.compute_encodings():
            output = quant_linear(input)

        input_enc = (quant_linear.input_quantizers[0].get_scale(),
                     quant_linear.input_quantizers[0].get_offset(),
                     quant_linear.input_quantizers[0].bitwidth)
        output_enc = (quant_linear.output_quantizers[0].get_scale(),
                      quant_linear.output_quantizers[0].get_offset(),
                      quant_linear.output_quantizers[0].bitwidth)
        weight_enc = (quant_linear.param_quantizers["weight"].get_scale(),
                      quant_linear.param_quantizers["weight"].get_offset(),
                      quant_linear.param_quantizers["weight"].bitwidth)
        weight_qdq = get_backend().quantize_dequantize(quant_linear.weight, *weight_enc, signed=True)
        output_expected = F.linear(input, weight_qdq, bias=quant_linear.bias)
        assert torch.equal(output, output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with an unquantized tensor
        Then: 1) output should be computed using the global true quant backend
              2) output should be a quantized tensor
              3) output should be close to fake quant output after dequantization
        """
        input_qdq = get_backend().quantize_dequantize(input, *input_enc)
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = get_backend().quantize_dequantize(output_fp, *output_enc)
        output_quant = quant_linear(input)
        assert isinstance(output_quant, DequantizedTensor)
        assert torch.allclose(output_quant.dequantize(), output_expected)

        """
        When: Invoke forward pass outside of compute_encodings context with a quantized tensor
        Then: Dequantized output should be close to running fake quant on the dequantized input tensor
        """
        quantized_input = affine_quantize(input, *input_enc)
        output = quant_linear(quantized_input)
        input_qdq = get_backend().dequantize(quantized_input, *input_enc[:2])
        output_fp = F.linear(input_qdq, weight_qdq, bias=quant_linear.bias)
        output_expected = get_backend().quantize_dequantize(output_fp, *output_enc)
        assert torch.allclose(output.dequantize(), output_expected)

    def test_no_input_quantizer(self, input):
        """
        Given: TrueQuantLinear with output and param quantizers and computed encodings
        """
        quant_linear = QuantizedLinear(10, input.shape[-1])
        quant_linear.output_quantizers[0] = Quantize((1, ), bitwidth=8, symmetric=False)
        quant_linear.param_quantizers["weight"] = Quantize((10, ), bitwidth=8, symmetric=True)
        with quant_linear.compute_encodings():
            quant_linear(input)
        """
        When: Invoke forward pass outside of compute_encodings with an unquantized tensor
        Then: raise RuntimeError
        """
        with pytest.raises(RuntimeError):
            quant_linear(input)

        """
        When: Invoke forward pass with a quantized tensor
        Then: return a tensor quantized with quant_linear.output_quantizer[0].encoding
        """
        quantizer = Quantize((1, ), bitwidth=8, symmetric=False)
        with quantizer.compute_encodings():
            quantizer(input)

        input_q = quantizer(input)
        output = quant_linear(input_q)
        assert isinstance(output, DequantizedTensor)
        assert output.encoding.scale == quant_linear.output_quantizers[0].get_scale()
        assert output.encoding.offset == quant_linear.output_quantizers[0].get_offset()


    def test_from_module(self, input):
        # Analogous to FakeQuantMixin.from_module test case
        """
        Given: Instantiate a true-quantized module using `TrueQuantMixin.from_module` and compute_encodings
        When: Inspect {input, output, param}_quantizers, they are the correct length
        """
        fp_linear = torch.nn.Linear(10, input.shape[-1])
        quant_linear = QuantizationMixin.from_module(fp_linear)

        assert len(quant_linear.input_quantizers) == 1
        assert len(quant_linear.output_quantizers) == 1
        assert len(quant_linear.param_quantizers) == 2

        """
        When: Inspect the parameters of the TrueQuant layer
        Then: They are identical to the parameters of the original layer
        """
        assert fp_linear.weight is quant_linear.weight
        assert fp_linear.bias is quant_linear.bias

        """
        When: Update to the parameter/buffer of the base FP module (or its submodule) using in-place operators.
              For example,
                1) fp_module.{param_or_buffer_name}.add_(1)
                2) fp_module.{submodule_name}.{param_or_buffer_name}.add_(1)
        Then: The result of in-place operation affects the parameters/buffers of the quantized module.
              In other words, the parameters/buffers of the quantized module will have been incremented by 1.
        """
        with torch.no_grad():
            fp_linear.weight.add_(1)
        assert torch.equal(fp_linear.weight, quant_linear.weight)
        with quant_linear.compute_encodings():
            quant_linear(input)

        """
        When: Reassign a new submodule/parameter/buffer to the base FP module using assignment stmt.
              For example,
                1) fp_module.{submodule_name} = torch.nn.Linear(...)
                2) fp_module.{param_or_buffer_name} = torch.empty(...)
        Then: The reassignment shouldn't affect the quantized module derived from the FP module.
              The vice versa should also hold.
        """
        fp_linear.weight = torch.nn.Parameter(torch.zeros(10, 10))
        assert not torch.all(fp_linear.weight == quant_linear.weight)


class TestQuantizedLayers:

    @pytest.mark.parametrize("layer", (torch.nn.Softmax(dim=1), torch.nn.Sigmoid(), torch.nn.GELU()))
    def test_layers_no_params(self, input, layer):
        fq_layer = fake_quant.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        fq_layer.input_quantizers[0] = QuantizeDequantize(shape=(1,), bitwidth=8, symmetric=False)
        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
        tq_layer.input_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)

        with fq_layer.compute_encodings():
            fq_layer(input)

        fq_output = fq_layer(input)


        with tq_layer.compute_encodings():
            tq_layer(input)
        tq_output = tq_layer(input)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.parametrize("layer", (torch.nn.Linear(10, 10), torch.nn.LayerNorm(10)))
    def test_layers_with_weight(self, input, layer):
        fq_layer = fake_quant.FakeQuantizationMixin.from_module(layer)
        tq_layer = QuantizationMixin.from_module(layer)
        fq_layer.input_quantizers[0] = QuantizeDequantize(shape=(1,), bitwidth=8, symmetric=False)
        fq_layer.output_quantizers[0] = QuantizeDequantize(shape=(1, ), bitwidth=8, symmetric=False)
        fq_layer.param_quantizers["weight"] = QuantizeDequantize(shape=(1,), bitwidth=8, symmetric=True)
        tq_layer.input_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)
        tq_layer.output_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)
        tq_layer.param_quantizers["weight"] = Quantize(shape=(1,), bitwidth=8, symmetric=True)

        with fq_layer.compute_encodings():
            fq_layer(input)

        fq_output = fq_layer(input)


        with tq_layer.compute_encodings():
            tq_layer(input)
        tq_output = tq_layer(input)

        assert torch.allclose(fq_output, tq_output.dequantize())

    @pytest.mark.cuda
    def test_layers_with_recompute(self):
        qlinear = QuantizedLinear(4096, 4096)
        qlinear.input_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)
        qlinear.output_quantizers[0] = Quantize(shape=(1,), bitwidth=8, symmetric=False)
        qlinear.param_quantizers["weight"] = Quantize(shape=(1,), bitwidth=8, symmetric=True)
        qlinear.cuda()

        # Using dummy backend is no good for testing memory saving in real life.
        # Set kernel to None so as to use FakeQuantizedLinear under the hood.
        qlinear.set_default_kernel(None)

        x = torch.randn((100, 4096), device="cuda:0")

        with qlinear.compute_encodings():
            qlinear(x)

        torch.cuda.empty_cache()
        with enable_recompute():
            out = qlinear(x)
        torch.cuda.synchronize()
        mem_with_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_with_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        del out

        torch.cuda.empty_cache()
        out = qlinear(x)
        torch.cuda.synchronize()
        mem_without_recompute = torch.cuda.memory_allocated()

        out.backward(torch.ones_like(out))
        grads_without_recompute = [param.grad.clone().detach().cpu() for param in qlinear.parameters()]
        for param in qlinear.parameters():
            param.grad = None

        # Expected memory saving:
        #   - Input quantizer save:
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        #   - Weight quantizer saves:
        #      - mask of shape [4096, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [4096, 4096] * 1 byte
        #   - F.linear saves:
        #      - quantized weight of shape [4096, 4096] * 4 bytes
        #      - quantized input of shape [100, 4096] * 4 bytes
        #   - Output quantizer saves:
        #      - linear output of shape [100, 4096] * 4 bytes
        #      - mask of shape [100, 4096] * 1 byte
        #      - quantized uint8 tensor of shape [100, 4096] * 1 byte
        expected_memory_saving = 0
        expected_memory_saving += (1 + 1) * x.numel() # input quantizer
        expected_memory_saving += (1 + 1) * qlinear.weight.numel() # weight quantizer
        expected_memory_saving += 4 * (qlinear.weight.numel() + x.numel()) # F.linear
        expected_memory_saving += (4 + 1 + 1) * out.numel() # output quantizer
        actual_memory_saving = mem_without_recompute - mem_with_recompute

        # Considering noise factors, actual memory saving should be no less than
        # 99% of the expected memory saving
        assert expected_memory_saving * 0.99 <= actual_memory_saving

        for grad_0, grad_1 in zip(grads_with_recompute, grads_without_recompute):
            assert torch.equal(grad_0, grad_1)