from code_generator.operators import place_holder
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.ConcatenationOptions import ConcatenationOptions

from .utils import get_input_tensors, get_nhwc_from_shape, get_output_tensors, getTensorTypeStr, is_constant


def parse_concatenation(op, model: Model.Model):
    """A CONCATENATION along the channels of any number of layer outputs that share H and W."""
    assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
    op_options = op.BuiltinOptions()
    concat_options = ConcatenationOptions()
    concat_options.Init(op_options.Bytes, op_options.Pos)

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]
    # the channels are the last axis, which a model may also name -1
    rank = output_tensor.tensor.ShapeAsNumpy().size
    assert concat_options.Axis() % rank == rank - 1, f"CONCATENATION along axis {concat_options.Axis()}, not the channels"
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    inputs = []
    for input_tensor in get_input_tensors(op, model):
        assert not is_constant(input_tensor), "CONCATENATION of a constant"
        _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
        assert (input_h, input_w) == (output_h, output_w), (
            f"CONCATENATION of a {input_h}x{input_w}x{input_c} input into a {output_h}x{output_w}x{output_c} output"
        )
        inputs.append({
            "idx": input_tensor.tensor_idx,
            "h": input_h,
            "w": input_w,
            "c": input_c,
            "dtype": getTensorTypeStr(input_tensor.tensor.Type()),
        })
    assert sum(t["c"] for t in inputs) == output_c, (
        f"CONCATENATION of {[t['c'] for t in inputs]} channels gives {output_c}"
    )
    dtype = getTensorTypeStr(output_tensor.tensor.Type())

    return place_holder.NInputPlaceHolderOp({
        "op": "CONCATENATION",
        "inputs": inputs,
        "output_idx": output_tensor.tensor_idx,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "input_dtype": dtype,
        "output_dtype": dtype,
        "color": "#ff66c4",  # pink
    })
