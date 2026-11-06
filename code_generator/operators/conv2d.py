import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts, op_inplace_type

__all__ = ["Conv2d"]

# USE_FP_REQ = True

default_params = {
    # op related
    "op": "CONV_2D",
    "is_SEBlock": False,
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
    "float_max": "FLT_MAX",
    "float_min": "-FLT_MAX",
    "padding": None,
    "padding_h": None,
    "padding_w": None,
    "dilation_h": None,
    "dilation_w": None,
    # fof Q training
    "need_Bmask": False,
    "output2_h": None,
    "output2_w": None,
    "output2_c": None,
    "output2_idx": None,
    "output2_dtype": "int8",
    # for partial channel update
    "first_k_channel": None,
    "inplace": op_inplace_type.flexible_inplace,
    "color": "#38b6ff", # turocoious blue
    "op_needs_right_shift": True
}


class Conv2d(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
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

    def set_input_zero_point(self, zero_x):
        p = self.params
        p["input_zero_point"] = zero_x

    def set_output_zero_point(self, zero_y):
        p = self.params
        p["input_zero_point"] = zero_y

    def get_macs(self):
        p = self.params
        return p["output_h"] * p["output_w"] * p["kernel_h"] * p["kernel_w"] * p["input_c"] * p["output_c"]

    def get_weights_size(self) -> int:
        p = self.params
        if p["input_dtype"] in {"float32", "fp32"}:
            size = 4
        else:
            size = 1
        return p["kernel_h"] * p["kernel_w"] * p["input_c"] * p["output_c"] * size

    def get_bias_size(self) -> int:
        p = self.params
        return 4 * p["output_c"]

    def get_scale_size(self) -> int:
        p = self.params
        return 4 * p["output_c"]

    def get_sbuf_size(self) -> int:
        params = self.params
        if params["input_dtype"] == params["output_dtype"] == "int8":
            p = self.params
            return p["kernel_h"] * p["kernel_w"] * p["input_c"] * 2 * 2  # 2 col and 16 bit
        else:
            return 0

    def get_kbuf_size(self) -> int:
        p = self.params
        if p["kernel_h"] == 1:
            return 0
        else:
            return self.get_weights_size() * 2  # 16bit
