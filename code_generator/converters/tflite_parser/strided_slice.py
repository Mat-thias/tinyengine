from code_generator.operators import place_holder
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.StridedSliceOptions import StridedSliceOptions

from .utils import get_input_tensors, get_nhwc_from_shape, get_np_from_wrapper, get_output_tensors, getTensorTypeStr


def parse_strided_slice(op, model: Model.Model, tmpPADIndice):
    """A STRIDED_SLICE cropping rows and columns off an NHWC tensor, as keras' Cropping2D does.

    NASNet pads a tensor by a row and a column at the end and crops as many off the start, which
    shifts it by a pixel. Like a conv or pool, the crop reads through such a PAD to the tensor before it.
    """
    assert op.BuiltinOptionsType() == BuiltinOptions.StridedSliceOptions
    op_options = op.BuiltinOptions()
    slice_options = StridedSliceOptions()
    slice_options.Init(op_options.Bytes, op_options.Pos)
    assert not (slice_options.EllipsisMask() or slice_options.NewAxisMask() or slice_options.ShrinkAxisMask()), (
        "STRIDED_SLICE with an ellipsis, a new axis or a shrunk axis is not a crop"
    )

    input_tensor, begin_tensor, end_tensor, strides_tensor = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    shape = input_tensor.tensor.ShapeAsNumpy().tolist()
    assert len(shape) == 4, f"STRIDED_SLICE of a {shape} tensor is not a crop of an NHWC one"
    strides = get_np_from_wrapper(strides_tensor).reshape(-1).tolist()
    assert strides == [1, 1, 1, 1], f"STRIDED_SLICE with strides {strides} skips elements, a crop keeps every one"
    # where each axis starts and stops: a masked axis is kept whole, a negative index counts from the end
    begin, end = [], []
    for axis, (b, e) in enumerate(zip(get_np_from_wrapper(begin_tensor).reshape(-1), get_np_from_wrapper(end_tensor).reshape(-1))):
        size = shape[axis]
        begin.append(0 if slice_options.BeginMask() >> axis & 1 else int(b) + size if b < 0 else int(b))
        end.append(size if slice_options.EndMask() >> axis & 1 else int(e) + size if e < 0 else min(int(e), size))
    assert (begin[0], end[0], begin[3], end[3]) == (0, shape[0], 0, shape[3]), (
        f"STRIDED_SLICE from {begin} to {end} of a {shape} tensor slices the batch or the channels"
    )

    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert (output_h, output_w, output_c) == (end[1] - begin[1], end[2] - begin[2], shape[3]), (
        f"STRIDED_SLICE from {begin} to {end} of a {shape} tensor gives {output_h}x{output_w}x{output_c}"
    )

    input_idx, input_h, input_w, input_c = input_tensor.tensor_idx, shape[1], shape[2], shape[3]
    fused_pad = (tmpPADIndice or {}).get(input_idx)
    if fused_pad is not None:
        input_idx = fused_pad.input_idx
        input_h -= fused_pad.pad_t + fused_pad.pad_b
        input_w -= fused_pad.pad_l + fused_pad.pad_r
    dtype = getTensorTypeStr(input_tensor.tensor.Type())

    return place_holder.SingleInputPlaceHolderOp({
        "op": "STRIDED_SLICE",
        "input_idx": input_idx,
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
        "color": "#b4a7d6",  # lavender
    })
