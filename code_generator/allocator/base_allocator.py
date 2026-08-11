# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   base_allocator.py
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

import matplotlib
import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from code_generator.constant import (
    FIGURE_CONFIG,
    TTYPE_INFERNECE,
    TTYPE_STATIC_WEIGHT,
    TTYPE_TRAINING_ACTIVATION,
    TTYPE_TRAINING_GRADIENT,
    TTYPE_TRAINING_WEIGHTS,
)


class BaseAllocator:
    def __init__(self, SRAM, sort_by_lifetime=False, allign_memory_32=False):
        self.rectangles = []
        self.SRAM = SRAM
        self.sort_by_lifetime = sort_by_lifetime
        self.allign_memory_32 = allign_memory_32
    
    # Description: add a tensor to schedule, return the index of the rectangle
    # Note: placement -1 indicates no placed yet
    def addTensor(
        self,
        start,
        end,
        size,
        placement=-1,
        name=None,
        type="activation",
        inplace=False,
        inplace_tensor_out_idx=None
    ) -> int:
        tensor_idx = len(self.rectangles)
        # Every rectangle is created plain, so we cannot know if it can be inplace except we
        # are sure that no other tensor needs it and we do this after all tensors are created
        assert not inplace and inplace_tensor_out_idx is None, \
            "inplace is decided by markInplace(), after every rectangle exists"

        self.rectangles.append(
            {
                "start": start,
                "end": end,
                "size": size,
                "placement": placement,
                "name": name,
                "type": type,
                "idx": tensor_idx,
                "inplace": inplace,
                "inplace_tensor_out_idx": inplace_tensor_out_idx,
                # Memory actually touched while this tensor is live. Same as "size"
                # for an ordinary tensor. This is to accomodate for layers that require
                # right shift for their inplace operations like convolution.
                "effective_size": size,
            }  # if this is set, we only need 1/4 of it after
        )
        return tensor_idx

    def markInplace(self, tensor_idx, out_idx, gap):
        """
        Record that the output rectangle `out_idx` is overwriting `tensor_idx`,
        so the two share one address.

        Called once every rectangle exists, because none of this is knowable at registration:
        the output is always registered after the tensor it overwrites, and its size is what
        decides how much of the buffer the pair really touches.
        """
        rec, out_rec = self.rectangles[tensor_idx], self.rectangles[out_idx]
        assert isinstance(out_idx, int) and out_idx >= 0, \
            f"the inplace output index must be a not negative integer, got {out_idx}"
        assert out_idx > tensor_idx, \
            f"the inplace output rectangle {out_idx} must be registered after the source tensor {tensor_idx}"
        assert not rec["inplace"], \
            f"rectangle {tensor_idx} is already given away to {rec['inplace_tensor_out_idx']}"

        rec["inplace"] = True
        rec["gap"] = gap
        rec["inplace_tensor_out_idx"] = out_idx
        # The output takes the buffer over the moment it starts being written, so
        # the tensor it overwrites dies exactly there, so it ends when the output starts.
        rec["end"] = out_rec["start"]
        assert rec["start"] <= rec["end"], \
            f"rectangle {tensor_idx} would die at {rec['end']} before it starts at {rec['start']}"
        # Shifted input at gap..gap+size, output written from the base, so the
        # workspace is whichever of the two reaches higher.
        rec["effective_size"] = max(rec["size"], gap + rec["size"], out_rec["size"])

    def getIdxAddress(self, idx):
        target_rec = None
        for rec in self.rectangles:
            if rec["idx"] == idx:
                target_rec = rec
        assert target_rec is not None
        return target_rec["placement"]

    def allocate(self):
        # place each rectangle
        print(f"Deriving the memory schedule for {len(self.rectangles)} activation tensors.")
        for cnt, rec in enumerate(tqdm(self.rectangles)):
            # fit each tensor into the memmory
            rec["placement"] = self.fit(rec)

    def fit(self, rec) -> int:  # memory address
        raise NotImplementedError

    def sortSize(self):
        sort_rectangles = []
        while len(self.rectangles) > 0:
            max_life = 0
            max_size = 0
            max_rectangle = None
            for rec in self.rectangles:
                if self.sort_by_lifetime:
                    life = rec["end"] - rec["start"]
                    if life > max_life:
                        max_life = life
                        max_rectangle = rec
                else:
                    if rec["size"] > max_size:
                        max_size = rec["size"]
                        max_rectangle = rec
            assert max_rectangle is not None
            sort_rectangles.append(max_rectangle)
            self.rectangles.remove(max_rectangle)
        self.rectangles = sort_rectangles

    def get_peak(self):
        peak = 0
        for rec in self.rectangles:
            # effective_size, not size: the buffer has to be large enough for the
            # right-inplace workspace (gap + input) too, otherwise the generated
            # buffer is short by exactly the amount the memmove overruns.
            rec_size = rec["placement"] + rec["effective_size"]
            if peak < rec_size:
                peak = rec_size
        return peak

    def visualize(self, path, train_start_idx=-1, scale=1024):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        max_y = max_x = 0

        for rec in self.rectangles:
            start, end, placement, size = (
                rec["start"],
                rec["end"],
                rec["placement"],
                rec["size"],
            )
            if max_y < rec["placement"] / scale + size / scale:
                max_y = rec["placement"] / scale + size / scale
            if max_x < end:
                max_x = end

        # x ticks
        max_x = max_x + (10 - max_x % 10)
        max_y = max_y + (10000 / scale - max_y % 10000 / scale)
        # x_ticks = numpy.arange(0, max_x + 1, step=max(1, int(max_x / 10)))
        x_ticks = numpy.arange(0, FIGURE_CONFIG["X_MAX"] + 1, step=FIGURE_CONFIG["X_STEP"])
        plt.xticks(x_ticks, fontsize=FIGURE_CONFIG["FONT_SIZE"])
        # y ticks
        y_ticks = numpy.arange(0, FIGURE_CONFIG["Y_MAX"] + 1, step=FIGURE_CONFIG["Y_STEP"])
        plt.yticks(y_ticks, fontsize=FIGURE_CONFIG["FONT_SIZE"])
        plt.xlim([0, FIGURE_CONFIG["X_MAX"]])
        plt.ylim([0, FIGURE_CONFIG["Y_MAX"]])
        plt.xlabel("Life cycle (operator)")
        plt.ylabel("Memory Footprint (KB)")
        plt.subplots_adjust(bottom=0.15)
        figure = plt.gcf()
        ax.title.set_fontsize(FIGURE_CONFIG["FONT_SIZE"])
        ax.xaxis.label.set_fontsize(FIGURE_CONFIG["FONT_SIZE"])
        ax.yaxis.label.set_fontsize(FIGURE_CONFIG["FONT_SIZE"])
        figure.set_size_inches(FIGURE_CONFIG["FIGURE_W_INCH"], FIGURE_CONFIG["FIGURE_H_INCH"])
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.patch.set_edgecolor("black")
        ax.patch.set_linewidth(2)

        for cnt, rec in enumerate(self.rectangles):
            start, end, placement, size = (
                rec["start"],
                rec["end"],
                rec["placement"],
                rec["size"],
            )
            hatch = None
            if rec["type"] == TTYPE_INFERNECE:
                color = FIGURE_CONFIG["INFERENCE_COLOR"]
            elif rec["type"] == TTYPE_TRAINING_WEIGHTS:
                color = FIGURE_CONFIG["TRAIN_WEIGHT_COLOR"]
            elif rec["type"] == TTYPE_TRAINING_ACTIVATION:
                color = FIGURE_CONFIG["TRAIN_ACTIVATION_COLOR"]
            elif rec["type"] == TTYPE_STATIC_WEIGHT:
                color = FIGURE_CONFIG["TRAIN_TENSOR_COLOR"]
            elif rec["type"] == TTYPE_TRAINING_GRADIENT:
                color = FIGURE_CONFIG["TRAIN_GRADIENT_COLOR"]
            else:
                raise NotImplementedError
            if rec["stride2_inplace_idx"]:
                # Draw the first one
                rect = matplotlib.patches.Rectangle(
                    (start, placement / scale),
                    rec["stride2_inplace_idx"] - start,
                    size / scale,
                    color=color,
                    hatch=hatch,
                )

                rect.set_edgecolor("black")
                ax.add_patch(rect)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                # Annotate index
                if FIGURE_CONFIG["SHOW_INDEX"]:
                    cx = (start + rec["stride2_inplace_idx"]) / 2
                    cy = (placement / scale) + (size / scale) / 2
                    ax.annotate(
                        str(rec["idx"]),
                        (cx, cy),
                        color="b",
                        fontsize=_get_index_font_size(
                            FIGURE_CONFIG["INDEX_FONT_SIZE"], (size / scale), FIGURE_CONFIG["Y_STEP"]
                        ),
                        weight="bold",
                        ha="center",
                        va="center",
                    )

                # Draw the 1/4 one
                rect = matplotlib.patches.Rectangle(
                    (rec["stride2_inplace_idx"], placement / scale),
                    end - rec["stride2_inplace_idx"],
                    int(size / scale / 4),
                    color=color,
                    hatch=hatch,
                )

                rect.set_edgecolor("black")
                ax.add_patch(rect)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                # Annotate index
                if FIGURE_CONFIG["SHOW_INDEX"]:
                    cx = (rec["stride2_inplace_idx"] + end) / 2
                    cy = (placement / scale) + (size / scale / 4) / 2
                    ax.annotate(
                        str(rec["idx"]),
                        (cx, cy),
                        color="b",
                        fontsize=_get_index_font_size(
                            FIGURE_CONFIG["INDEX_FONT_SIZE"], (size / scale), FIGURE_CONFIG["Y_STEP"]
                        ),
                        weight="bold",
                        ha="center",
                        va="center",
                    )
            else:
                rect = matplotlib.patches.Rectangle(
                    (start, placement / scale),
                    end - start,
                    size / scale,
                    color=color,
                    hatch=hatch,
                )

                rect.set_edgecolor("black")
                ax.add_patch(rect)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # Annotate index
                if FIGURE_CONFIG["SHOW_INDEX"]:
                    cx = (start + end) / 2
                    cy = (placement / scale) + (size / scale) / 2
                    ax.annotate(
                        str(rec["idx"]),
                        (cx, cy),
                        color="b",
                        fontsize=_get_index_font_size(
                            FIGURE_CONFIG["INDEX_FONT_SIZE"], (size / scale), FIGURE_CONFIG["Y_STEP"]
                        ),
                        weight="bold",
                        ha="center",
                        va="center",
                    )

        plt.savefig(path, dpi=FIGURE_CONFIG["DPI"])


def _get_index_font_size(origin_font_size, y_size, y_block_size):
    y_bound = y_block_size / 4
    if y_size > y_bound:
        return origin_font_size
    else:
        return int(origin_font_size * (y_size / y_bound))
