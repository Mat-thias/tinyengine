from code_generator.operators import maxpool2d
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.Pool2DOptions import Pool2DOptions

from .utils import fold_padding, get_input_tensors, get_output_tensors


def parse_maxpool(op, model: Model.Model, tmpPADIndice=None):
    # Incase no params
    input_type = None
    input_zero_point = None
    output_zero_point = None
    input_scale = None
    output_scale = None

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 1, "input tensors length should be 1"

    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # shapes
    _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
    _, output_h, output_w, output_c = output_tensor.tensor.ShapeAsNumpy()

    # pool parameters
    assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
    op_options = op.BuiltinOptions()
    pool2d_options = Pool2DOptions()
    pool2d_options.Init(op_options.Bytes, op_options.Pos)
    stride_h = pool2d_options.StrideH()
    stride_w = pool2d_options.StrideW()
    filter_h = pool2d_options.FilterHeight()
    filter_w = pool2d_options.FilterWidth()

    # padding, and the tensor the pool really reads if a PAD is folded into it
    input_idx, input_h, input_w, (pad_t, pad_b, pad_l, pad_r) = fold_padding(
        "MAX_POOL_2D", pool2d_options.Padding(), input_tensor,
        (filter_h, filter_w), (stride_h, stride_w), (output_h, output_w), tmpPADIndice,
    )

    pool_params = {
        # operator
        "op": "MAX_POOL_2D",
        # pool parameters
        "filter_h": filter_h,
        "filter_w": filter_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "pad_t": pad_t,
        "pad_b": pad_b,
        "pad_l": pad_l,
        "pad_r": pad_r,
        # tensor
        "input_idx": input_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dim": 3,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtype": input_type,
        # trainable parameters
        "input_zero_point": input_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "output_scale": output_scale,
    }
    op = maxpool2d.maxPool2d(pool_params)

    return op
