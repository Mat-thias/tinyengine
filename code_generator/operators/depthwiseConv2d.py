import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts, op_inplace_type

__all__ = ["DepthwiseConv2d"]

default_params = {
    # op related
    "op": "DEPTHWISE_CONV_2D",
    "is_SEBlock": False,
    # one value per side of the input: top, bottom, left, right
    "pad_t": None,
    "pad_b": None,
    "pad_l": None,
    "pad_r": None,
    "stride_h": None,
    "stride_w": None,
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "dtypte": "int8",
    "kernel_h": None,
    "kernel_w": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # quantization related
    "weight_value": None,
    "weight_name": None,
    "bias": None,
    "bias_name": None,
    "effective_scale": None,
    "input_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "output_scale": None,
    "weight_scale": None,
    "multiplier": None,
    "shift": None,
    # for fp implementation
    "dilation_h": None,
    "dilation_w": None,
    "float_max": "FLT_MAX",
    "float_min": "-FLT_MAX",
    # fof Q training
    "need_Bmask": False,
    "output2_h": None,
    "output2_w": None,
    "output2_c": None,
    "output2_idx": None,
    "output2_dtype": "int8",
    "inplace": op_inplace_type.force_inplace,
    "color": "#ff914d", # orange
    "op_needs_right_shift": False
}


class DepthwiseConv2d(basicOperator):
    def __init__(self, params: dict, USE_INPLACE: bool = True) -> None:
        self.params = deep_copy_dicts(default_params)
        self.USE_INPLACE = USE_INPLACE
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self):
        p = self.params
        return p["output_h"] * p["output_w"] * p["kernel_h"] * p["kernel_w"] * p["input_c"]

    def get_activation_size(self) -> int:
        size = 0
        for t in self.input_tensors:
            size += t.len()
        return size

    def get_weights_size(self) -> int:
        p = self.params
        return p["kernel_h"] * p["kernel_w"] * p["input_c"]

    def get_bias_size(self) -> int:
        p = self.params
        return 4 * p["output_c"]

    def get_scale_size(self) -> int:
        p = self.params
        return 4 * p["output_c"]

    def get_sbuf_size(self) -> int:
        p = self.params
        if self.USE_INPLACE:
            return 2 * (p["input_h"] + p["pad_t"] + p["pad_b"]) * (p["input_w"] + p["pad_l"] + p["pad_r"])  # 2 x resolution
        else:
            return 2 * p["kernel_h"] * p["kernel_w"] * p["input_c"]  # 16 bit

    def get_kbuf_size(self) -> int:
        return 0
