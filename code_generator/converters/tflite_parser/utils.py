import logging
import math

import numpy as np

from code_generator.tflite import Model
from code_generator.tflite.BuiltinOperator import BuiltinOperator
from code_generator.tflite.Padding import Padding
from code_generator.tflite.TensorType import TensorType


class TFLiteTensorWrpper:
    def __init__(self, tensor_idx, tensor, buffer, qnn_params):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


def getOpCodeStr(op, model: Model.Model):
    op_code_list_idx = op.OpcodeIndex()
    op_code_id = model.OperatorCodes(op_code_list_idx).DeprecatedBuiltinCode()

    def _build_str_map(obj):
        ret = {}
        for field_name in dir(obj):
            if not field_name.startswith("_"):
                field_value = getattr(obj, field_name)
                if isinstance(field_value, int):
                    ret[field_value] = field_name
        return ret

    builtin_op_code = _build_str_map(BuiltinOperator())

    return builtin_op_code[op_code_id]


def get_np_from_wrapper(wrapper):
    if wrapper.tensor.Type() == TensorType.INT8:
        dtype = np.int8
    elif wrapper.tensor.Type() == TensorType.INT32:
        dtype = np.int32
    elif wrapper.tensor.Type() == TensorType.FLOAT32:
        dtype = np.float32
        logging.warn("Support of floating-point tensors are experimental.")
    else:
        raise NotImplementedError("Current implementation only supports int8 and int32")

    data = wrapper.buffer.DataAsNumpy()
    shape = wrapper.tensor.ShapeAsNumpy() if wrapper.tensor.ShapeLength() != 0 else []

    return np.frombuffer(data, dtype=dtype).reshape(shape)


def is_constant(wrapper):
    """True for a tensor the model stores, such as a weight, rather than one a layer produces."""
    return wrapper.buffer.DataLength() > 0


def get_input_tensors(op, model: Model.Model):
    return _get_wrapper_tensors(op.InputsAsNumpy(), model)


def get_output_tensors(op, model: Model.Model):
    return _get_wrapper_tensors(op.OutputsAsNumpy(), model)


def _get_wrapper_tensors(tensor_index_list, model: Model.Model):
    ret = []
    subgraph = model.Subgraphs(0)
    for idx in tensor_index_list:
        tensor = subgraph.Tensors(idx)
        buffer_idx = tensor.Buffer()
        buffer = model.Buffers(buffer_idx)

        tflite_qparams = tensor.Quantization()

        qparams_to_tensor_wrapper = None
        if tflite_qparams:
            scale = tflite_qparams.ScaleAsNumpy()
            zero_point = tflite_qparams.ZeroPointAsNumpy()
            if isinstance(zero_point, np.ndarray):
                # Per-channel quantization
                if scale.size != 1 and zero_point.size != 1:
                    qparams_to_tensor_wrapper = {"scale": scale, "zero_point": zero_point}
                # Per-tensor quantization
                elif scale.size == 1 and zero_point.size == 1:
                    qparams_to_tensor_wrapper = {"scale": float(scale[0]), "zero_point": int(zero_point[0])}
                else:
                    raise NotImplementedError
            elif scale == zero_point == 0:
                pass
        else:
            logging.warn("Quantization parameters not found in the model! Floating-point opeartos are experimental.")

        ret.append(TFLiteTensorWrpper(idx, tensor, buffer, qparams_to_tensor_wrapper))
    return ret


def getLayerMultiplierShift(effective_scale):
    if effective_scale == 0:
        significand = 0
        shift = 0
    else:
        sig, shi = math.frexp(effective_scale)
        sig = int(round(sig * 2**31))

        if sig == 2**31:
            sig /= 2
            shi += 1
        if shi < -31:
            shi = 0
            sig = 0

        significand = sig
        shift = shi

    return significand, shift


def getMultiplierShift(effective_scale):
    significand = np.zeros(len(effective_scale), dtype="int32")
    shift = np.zeros(len(effective_scale), dtype="int32")

    for i, s in enumerate(effective_scale):
        if s == 0:
            significand[i] = 0
            shift[i] = 0
        else:
            sig, shi = math.frexp(s)
            sig = int(round(sig * 2**31))

            if sig == 2**31:
                sig /= 2
                shi += 1
            if shi < -31:
                shi = 0
                sig = 0

            significand[i] = sig
            shift[i] = shi

    return significand, shift


def getTensorTypeStr(type):
    if TensorType.INT8 == type:
        return "int8"
    if TensorType.UINT8 == type:
        return "uint8"
    if TensorType.FLOAT32 == type:
        return "float32"


def get_hwc_from_chwshape(shape):
    h = 1
    w = 1
    c = 1
    if len(shape) == 4:
        c = shape[1]
        h = shape[2]
        w = shape[3]
    elif len(shape) == 3:
        c = shape[1]
        h = shape[2]
    elif len(shape) == 2:
        c = shape[1]
    return h, w, c


def get_nhwc_from_shape(shape):
    n = 1
    h = 1
    w = 1
    c = 1
    if len(shape) == 4:
        c = shape[3]
        w = shape[2]
        h = shape[1]
    elif len(shape) == 3:
        c = shape[2]
        w = shape[1]
        h = shape[0]
    elif len(shape) == 2:
        c = shape[1]
        w = shape[0]
    elif len(shape) == 1:
        c = shape[0]
    return n, h, w, c


def fold_padding(op_name, padding_mode, input_tensor, window, stride, output_hw, tmpPADIndice):
    """The tensor a windowed op really reads, its extent, and the padding it adds to each side.

    A SAME op is padded the way tflite sizes it. A VALID op behind a PAD takes that PAD's padding
    and reads the tensor before it, which is smaller by the padding. A VALID op with no PAD is not
    padded. window, stride and output_hw are (h, w) pairs.

    Returns (input_idx, input_h, input_w, (pad_t, pad_b, pad_l, pad_r)).
    """
    _, input_h, input_w, _ = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    input_idx = input_tensor.tensor_idx
    fused_pad = (tmpPADIndice or {}).get(input_idx)
    if padding_mode == Padding.SAME:
        assert fused_pad is None, f"a PAD before the SAME {op_name} would pad its input twice"
        pad_t, pad_b = _same_padding(input_h, window[0], stride[0], output_hw[0])
        pad_l, pad_r = _same_padding(input_w, window[1], stride[1], output_hw[1])
    elif fused_pad is not None:
        input_idx = fused_pad.input_idx
        pad_t, pad_b, pad_l, pad_r = fused_pad.pad_t, fused_pad.pad_b, fused_pad.pad_l, fused_pad.pad_r
        input_h = input_h - (pad_t + pad_b)
        input_w = input_w - (pad_l + pad_r)
    else:
        pad_t = pad_b = pad_l = pad_r = 0
    # a pad as wide as the window would leave an output reading nothing but padding
    assert max(pad_t, pad_b) < window[0] and max(pad_l, pad_r) < window[1], (
        f"padding {(pad_t, pad_b, pad_l, pad_r)} against the {window[0]}x{window[1]} window of {op_name}"
    )
    return input_idx, input_h, input_w, (pad_t, pad_b, pad_l, pad_r)


def _same_padding(extent, window, stride, output):
    """The padding SAME puts before and after one axis.

    SAME sizes the output at ceil(extent / stride) and pads by however far the last window
    overruns the input. tflite puts the odd row of an odd total after, so a stride 2 3x3 conv
    over an even extent pads (0, 1).
    """
    total = max((output - 1) * stride + window - extent, 0)
    return total // 2, total - total // 2
