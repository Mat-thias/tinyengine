# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   mcunet_inx.py
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, busoyeopegbemimatthias@gmail.com
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

"""
Codegen entry point for the MCUNet-inX image classifiers
============================================================

Generates the C source for one of mcunet-in0 .. mcunet-in4 and reports the
peak activation memory the schedule needs.

Usage
-----
    python examples/mcunet_inx.py -x 4

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
from mcunet.mcunet.model_zoo import download_tflite

parser = argparse.ArgumentParser("mcunet")
parser.add_argument(
    "-x", type=int, required=True, choices=list(range(0, 5)),
    help="specifies the mcunet_inx model to work with"
)
args = parser.parse_args()

# Fetched once and cached by the model zoo; later runs are offline.
tflite_path = download_tflite(net_id=f"mcunet-in{args.x}")

# Parses the tflite, schedules the activations (MILP), and emits the C source.
# life_cycle_path is the allocation figure, not a model artefact.
peakmem = GenerateSourceFilesFromTFlite(
    tflite_path,
    life_cycle_path="./lifecycle.png",
)

# Peak activation memory only: the weights live in flash and are not counted.
print(f"Peak memory: {peakmem} bytes")
