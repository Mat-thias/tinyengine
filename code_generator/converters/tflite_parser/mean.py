from code_generator.operators import avgpool2d
from code_generator.tflite import Model

from .utils import get_input_tensors, get_nhwc_from_shape, get_np_from_wrapper, get_output_tensors, getTensorTypeStr


def mean_axes(op, model: Model.Model):
    """The axes a MEAN reduces, counted from the front of its input."""
    input_tensor, axes_tensor = get_input_tensors(op, model)
    rank = input_tensor.tensor.ShapeAsNumpy().size
    return {int(axis) % rank for axis in get_np_from_wrapper(axes_tensor).reshape(-1)}


def parse_mean(op, model: Model.Model):
    """A MEAN over H and W, which is a global average pool: the layer mean1dto2d builds for a MEAN pair."""
    assert mean_axes(op, model) == {1, 2}, f"MEAN over axes {sorted(mean_axes(op, model))} is not over H and W"
    input_tensor = get_input_tensors(op, model)[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    # (N, C), or (N, 1, 1, C) when the MEAN keeps its dimensions
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert output_h == output_w == 1 and output_c == input_c, (
        f"MEAN over H and W of a {input_h}x{input_w}x{input_c} input gives {output_h}x{output_w}x{output_c}"
    )
    dtype = getTensorTypeStr(input_tensor.tensor.Type())

    params = {
        # operator
        "op": "AVERAGE_POOL_2D",
        # one window over the whole plane
        "filter_h": input_h,
        "filter_w": input_w,
        "stride_h": 1,
        "stride_w": 1,
        "pad_t": 0,
        "pad_b": 0,
        "pad_l": 0,
        "pad_r": 0,
        # tensor
        "input_idx": input_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dim": 3,
        "output_dim": 3,
        "output_h": 1,
        "output_w": 1,
        "output_c": output_c,
        "input_dtype": dtype,
        "output_dtype": dtype,
    }

    return avgpool2d.AvgPool2d(params)
