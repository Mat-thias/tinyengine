from code_generator.operators import place_holder
from code_generator.tflite import Model

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getOpCodeStr, getTensorTypeStr

# activations that read one tensor and write one of the same shape
ACTIVATIONS = ("HARD_SWISH", "LOGISTIC", "RELU")


def parse_activation(op, model: Model.Model):
    op_name = getOpCodeStr(op, model)
    assert op_name in ACTIVATIONS, f"{op_name} is not one of {ACTIVATIONS}"

    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 1, f"{op_name} reads {len(input_tensors)} tensors"
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert (output_h, output_w, output_c) == (input_h, input_w, input_c), (
        f"{op_name} of {input_h}x{input_w}x{input_c} gives {output_h}x{output_w}x{output_c}"
    )
    dtype = getTensorTypeStr(input_tensor.tensor.Type())

    return place_holder.SingleInputPlaceHolderOp({
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
        "color": "#d9d9d9",  # grey
    })
