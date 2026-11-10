from code_generator.operators import place_holder
from code_generator.tflite import Model

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getOpCodeStr, getTensorTypeStr, is_constant

# colours in the lifecycle plot
COLORS = {"ADD": "#78d957", "SUB": "#78d957", "MUL": "#c9e265"}  # green, green, yellow green


def parse_elementwise(op, model: Model.Model):
    """ADD, SUB or MUL by a tensor of the same shape, by one value per channel, or by a single value.

    A layer produces the first operand and the output has its shape. The second is either another
    layer's output, read as a second input, or a constant the model stores, which is a weight and
    so not scheduled.
    """
    op_name = getOpCodeStr(op, model)
    assert op_name in COLORS, f"{op_name} is not an element-wise ADD, SUB or MUL"
    input_tensor, input2_tensor = get_input_tensors(op, model)

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert (output_h, output_w, output_c) == (input_h, input_w, input_c), (
        f"{op_name} of {input_h}x{input_w}x{input_c} gives {output_h}x{output_w}x{output_c}"
    )
    # the second operand matches the first, or is broadcast over H and W, or over everything
    assert (input2_h, input2_w, input2_c) in ((input_h, input_w, input_c), (1, 1, input_c), (1, 1, 1)), (
        f"{op_name} of {input_h}x{input_w}x{input_c} by {input2_h}x{input2_w}x{input2_c}"
    )
    assert not is_constant(input_tensor), f"{op_name} with its constant operand first"
    dtype = getTensorTypeStr(input_tensor.tensor.Type())

    params = {
        "op": op_name,
        "input_idx": input_tensor.tensor_idx,
        "input_dim": 3,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "output_idx": output_tensor.tensor_idx,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "input_dtype": dtype,
        "output_dtype": dtype,
        "color": COLORS[op_name],
    }
    if is_constant(input2_tensor):
        return place_holder.SingleInputPlaceHolderOp(params)

    params.update({
        "input2_idx": input2_tensor.tensor_idx,
        "input2_dim": 3,
        "input2_h": input2_h,
        "input2_w": input2_w,
        "input2_c": input2_c,
        "input2_dtype": getTensorTypeStr(input2_tensor.tensor.Type()),
    })
    return place_holder.DoubleInputPlaceHolderOp(params)
