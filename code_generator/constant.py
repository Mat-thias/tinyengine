# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   constant.py
#
# Reference papers:
#  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
#  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
#  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
# Contact authors:
#  - Wei-Ming Chen, wmchen@mit.edu
#  - Wei-Chen Wang, wweichen@mit.edu
#  - Ji Lin, jilin@mit.edu
#  - Ligeng Zhu, ligeng@mit.edu
#  - Song Han, songhan@mit.edu
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

TTYPE_INFERNECE = "inference"

SKIP_OPs = {"QUANTIZE", "DEQUANTIZE"}

FUSE_SGD_UPDATE_STR = "INPLACE_SGD"
FUSHION_CONFIG = {
    FUSE_SGD_UPDATE_STR: True,
}
INFERECE_WEIGHT_SIZE = 1
# end: for forcelly cast to fp32
TTYPE_STATIC_WEIGHT = "static_weights"
TTYPE_TRAINING_GRADIENT = "training_gradient"
TTYPE_TRAINING_ACTIVATION = "training_activation"
TTYPE_TRAINING_WEIGHTS = "training_weight"
TTYPE_INFERNECE = "inference"

# template 3
FIGURE_CONFIG = {
    "TRAIN_TENSOR_COLOR": "#68ac14",  # green
    "TRAIN_WEIGHT_COLOR": "#61a78f",  # light green:
    "TRAIN_ACTIVATION_COLOR": "#b9c8d8",  # light blue
    "TRAIN_GRADIENT_COLOR": "#35619f",  # blue
    "INFERENCE_COLOR": "#eab11f",  # yellow
    "FIGURE_W_INCH": 19,
    "FIGURE_H_INCH": 7,
    "DPI": 800,
    "Y_STEP": 96,
    "Y_MAX": 384,
    "X_STEP": 30,
    "X_MAX": 270,
    "SHOW_INDEX": False,
    "FONT_SIZE": 32,
    "INDEX_FONT_SIZE": 24,
}
