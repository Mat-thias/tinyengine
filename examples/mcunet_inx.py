# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   mcunet_inx.py
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
#  - Tolulope Matthew Busoye, matthew@powerlabstech.com
#  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

"""
Codegen entry point for the models in the MCUNet zoo
============================================================

Generates the C source for one released model and reports the peak activation
memory the schedule needs.

Usage
-----
    python examples/mcunet_inx.py -x 4              # mcunet-in4
    python examples/mcunet_inx.py -n mcunet-vww1    # any other zoo model

-x is shorthand for the mcunet-inX classifiers, which is what the sweep in
mcunet_comparison_results.py reaches for most often; -n takes any id in
net_id_list, so the vww, mbv2 and proxyless releases go through the same path.

Run from the repository root: the paths below are relative to the working
directory, not to this file.

Writes
------
    codegen/Source/genModel.c        the invoke() call sequence
    codegen/Include/genModel.h       weights, scales, and the activation buffer
    codegen/Include/genModelShape.h  input / output shapes, for the harness
    codegen/Source/depthwise_*.c     the depthwise kernels this model needs
    ./lifecycle.png                  tensor lifetimes against memory address

The generated files are overwritten on every run, so codegen/ only ever holds
the most recently generated model. Both harnesses (deploy/pc and deploy_pico2)
compile against whatever is there, which is why they read the shapes from
genModelShape.h rather than hardcoding one model's resolution.
"""

import sys

# "." puts the repo root first so `code_generator` resolves to this checkout
# rather than any installed copy; ".." reaches the sibling `mcunet` package.
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import argparse

from code_generator.CodegenUtilTFlite import GenerateSourceFilesFromTFlite
from mcunet.mcunet.model_zoo import download_tflite, net_id_list

parser = argparse.ArgumentParser("mcunet")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-x", type=int, choices=list(range(0, 5)),
    help="specifies the mcunet_inx model to work with"
)
group.add_argument(
    "-n", "--net-id", choices=net_id_list,
    help="any model in the zoo, for the releases -x cannot name"
)
parser.add_argument(
    "-f", "--life_cycle_file", type=str, default="./lifecycle.png",
    help="the location to save the tensor lifecycle visualization"
)
parser.add_argument(
    "-d", "--disable_inplace_option", default=False, action="store_true",
    help="to return to dual buffer execution"
)
args = parser.parse_args()

net_id = args.net_id or f"mcunet-in{args.x}"

# Fetched once and cached by the model zoo; later runs are offline.
tflite_path = download_tflite(net_id=net_id)

# Parses the tflite, schedules the activations (MILP), and emits the C source.
# life_cycle_path is the allocation figure, not a model artefact.
peakmem, _ = GenerateSourceFilesFromTFlite(
    tflite_path,
    model_name=net_id,
    life_cycle_path=args.life_cycle_file,
    disable_inplace_option = args.disable_inplace_option
)

# Peak activation memory only: the weights live in flash and are not counted.
print(f"Peak memory: {peakmem} bytes")
