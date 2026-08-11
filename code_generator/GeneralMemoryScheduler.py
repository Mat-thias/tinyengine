# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   GeneralMemoryScheduler.py
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

from .allocator.milp_allocator import MILPAllocator

from .operators.basic_utils import allign_byte_to_n
from .constant import (
    FUSE_SGD_UPDATE_STR,
    FUSHION_CONFIG,
    INFERECE_WEIGHT_SIZE,
    TTYPE_INFERNECE,
    TTYPE_STATIC_WEIGHT,
    TTYPE_TRAINING_ACTIVATION,
    TTYPE_TRAINING_GRADIENT,
    TTYPE_TRAINING_WEIGHTS,
)

from .operators import add, avgpool2d, conv2d, depthwiseConv2d
from .operators.right_inplace import right_inplace_gap_unpadded_ext

# NOTE: urrently only supports ops a single output
INPLACE_OPS = {
    0: (),
    1: (
        avgpool2d.AvgPool2d,
        conv2d.Conv2d,
        depthwiseConv2d.DepthwiseConv2d,
    ),
    2: (
        add.Add,
    )
}

class GeneralMemoryScheduler:
    def __init__(
        self,
        layer,
        tflite_op=False,
        dummy_address=False,
        memory_limit=10 * 1024 * 1024,
        inplace=True,
        outputTables=None,
        mem_visual_path="codegen/allocation.png",
        VisaulizeTrainable=True,
        sort_by_lifetime=False,
        allign_memory_32=False
    ):
        self.layer = layer
        self.heads = 0
        self.buffers = {
            "input_output": 0,
            "residual": 0,
            "im2col": 0,
            "kernel": 0,
            "feature": 0,
            "trainable": 0,
        }  # for feature pyramid
        # overall memory info
        self.peakmem = 0
        self.flash = 0
        self.bias = 0
        self.scale = 0
        self.code = 0
        self.allocator = MILPAllocator(memory_limit, sort_by_lifetime, allign_memory_32)
        self.outputTables = outputTables
        self.USE_INPLACE = inplace
        self.mem_visual_path = mem_visual_path
        self.tflite_op = tflite_op
        self.dummy_address = dummy_address
        self.VisaulizeTrainable = VisaulizeTrainable
        self.allign_memory_32 = allign_memory_32

        # for showing layer-wise memory usage
        self.layermem = []

    def _isTranable(self, name):
        for o in self.outputTables:
            if isinstance(name, str) and o.name in name:
                return True
        return False

    def allocateMemory(self):
        # NOTE: I am not sure what this section of code does, but it is never activate as
        # self.outputTables is always None in my test, so I will leave it as is for now.
        assert self.outputTables is None or (isinstance(self.outputTables, list) and len(self.outputTables) == 0), "outputTables is must be None or a list for this version of the code"
        num_layers = len(self.layer)
        # add all trainable tensors as one tensor
        length_model = len(self.layer)
        trainable = 0
        weight_size = 0
        bias_size = 0
        for out_t in self.outputTables:
            if "bias" in out_t.name:
                dtype_multiplier = 4
                trainable += int(out_t.len * dtype_multiplier)
                bias_size += int(out_t.len * dtype_multiplier)
            elif "weight" in out_t.name:
                dtype_multiplier = INFERECE_WEIGHT_SIZE
                # find the conv2d owning the tensor
                conv_2d_op = None
                for lay in self.layer:
                    if "weight_name" in lay.params and out_t.name in lay.params["weight_name"]:
                        conv_2d_op = lay
                        break
                assert conv_2d_op is not None
                # check if it is partial
                if "first_k_channel" in conv_2d_op.params and conv_2d_op.params["first_k_channel"] is not None:
                    trainable += int(
                        out_t.len
                        * dtype_multiplier
                        * conv_2d_op.params["first_k_channel"]
                        / conv_2d_op.params["input_c"]
                    )
                    weight_size += int(
                        out_t.len
                        * dtype_multiplier
                        * conv_2d_op.params["first_k_channel"]
                        / conv_2d_op.params["input_c"]
                    )
                else:
                    trainable += int(out_t.len * dtype_multiplier)
                    weight_size += int(out_t.len * dtype_multiplier)
            else:
                pass
        assert not self.VisaulizeTrainable, f"Training is currently not supported this disables that."
        if self.VisaulizeTrainable:
            self.allocator.addTensor(0, length_model, trainable, type=TTYPE_STATIC_WEIGHT)

        def is_layer_last_consumer_of_tensor(l_idx, t):
            """Returns True if not succeding layer of l consumes tensor t"""
            for idx in range(l_idx+1, len(self.layer)):
                for input_t in self.layer[idx].input_tensors:
                    if str(t.graph_idx) == str(input_t.graph_idx):
                        return False
            return True

        def tensor_inplace_parameters_in(t, inplace, gap, t_outs):
            """set the 3 parameters needed by input tensorfor inplace operations and memory scheduling"""
            t.inplace = inplace
            t.gap = allign_byte_to_n(gap, (32 if self.allign_memory_32 else 4))
            t.inplace_tensor_out = t_outs[0]

        # If op is an inplace op and input is not input to another op; is the input is
        # only used by this op, the we set it to be inplace to overwrite the input esle false
        # so we have to scan through the whole graph to check if each op *(first)* input
        # isn't an input to another op so we don't overwrite
        for i, op in enumerate(self.layer):
            if isinstance(op, INPLACE_OPS[1]):
                t = op.input_tensors[0]
                gap = 0
                inplace = is_layer_last_consumer_of_tensor(i, t)
                if isinstance(op, conv2d.Conv2d):
                    # adding gap to the output tensor only for conv2d,
                    # to cater for the inplace convolution head room
                        params = op.params
                        gap = right_inplace_gap_unpadded_ext(
                            cin=params['input_c'], cout=params['output_c'],
                            k_h=params['kernel_h'], k_w=params['kernel_w'],
                            hin=params['input_h'], win=params['input_w'],
                            s_h=params['stride_h'], s_w=params['stride_w'],
                            d_h=params['dilation_h'], d_w=params['dilation_w'],
                            p_h=params['padding_h'], p_w=params['padding_w']
                        )
                # alligning the memory so the memmove operation is not fragmented which speeds it up
                tensor_inplace_parameters_in(t, inplace, gap, op.output_tensors)

            elif isinstance(op, INPLACE_OPS[2]):
                t1, t2 = op.input_tensors
                inplace1 = is_layer_last_consumer_of_tensor(i, t1)
                inplace2 = is_layer_last_consumer_of_tensor(i, t2)
                gap1 = gap2 = 0
                # alligning the memory so the memmove operation is not fragmented which speeds it up
                tensor_inplace_parameters_in(t1, inplace1, gap1, op.output_tensors)
                tensor_inplace_parameters_in(t2, inplace2, gap2, op.output_tensors)

            else:
                raise AttributeError(f"Unaccounted Layerr {op} {op.params['op']}")

        all_t_size = 0
        # go through all tensors in the model
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            unallocated_tensors = []
            for t in op.input_tensors:
                if t.allocator_idx is None:
                    unallocated_tensors.append(t)
            for cnt, t in enumerate(op.output_tensors):
                if t.allocator_idx is None:
                    unallocated_tensors.append(t)

            # add each tensor
            training_start_idx = _find_training_idx(layers=self.layer)
            assert training_start_idx == len(self.layer), \
                f"The current implementation does not support traning yet, this is a guide for that"
            for cnt, t in enumerate(unallocated_tensors):
                t.allign_memory_32 = self.allign_memory_32
                start_idx = i
                # TODO: this is temp solution
                if training_start_idx > i and "out_multiply" not in t.graph_idx:
                    end_idx = i + 1 if i == 0 else num_layers
                else:
                    end_idx = i + 1
                for idx in range(start_idx + 1, num_layers):
                    for input_t in self.layer[idx].input_tensors:
                        if str(t.graph_idx) == str(input_t.graph_idx):
                            end_idx = idx + 1
                assert isinstance(end_idx, int), f"end_idx is not set for tensor {t.graph_idx} in layer {i}"
                # check if this is output
                ttype = TTYPE_INFERNECE
                if self.outputTables is not None and not FUSHION_CONFIG[FUSE_SGD_UPDATE_STR]:
                    for o in self.outputTables:
                        if o.idx in t.graph_idx:
                            end_idx = len(self.layer)
                            all_t_size += o.len
                            ttype = TTYPE_TRAINING_GRADIENT

                # for patchbased inference, we need the input tensro to be allocated in the patch inference stage
                if (
                    "is_start_of_normal_inference_block" in op.params
                    and op.params["is_start_of_normal_inference_block"]
                ):
                    if t in op.input_tensors:
                        start_idx = 0
                # add the tensor
                t.allocator_idx = self.allocator.addTensor(
                    start_idx, end_idx, t.len(), name=t.graph_idx, type=ttype
                )
                # propagate the allocation to tensors with the same idx
                for j in range(i + 1, num_layers):
                    opp = self.layer[j]
                    for tt in opp.input_tensors:
                        if str(t.graph_idx) == str(tt.graph_idx):
                            tt.allocator_idx = t.allocator_idx
                    # not inplace update
                    for tt in opp.output_tensors:
                        if str(t.graph_idx) == str(tt.graph_idx):
                            tt.allocator_idx = t.allocator_idx

            # for detailed memory
            layermem = {}

            layermem["MAC"] = op.get_macs()
            layermem["activation"] = op.get_activation_size()
            layermem["scale"] = op.get_scale_size()
            layermem["runtime"] = op.get_sbuf_size()
            layermem["kernel"] = op.get_kbuf_size()
            self._enlargeBuffer("im2col", layermem["runtime"])
            self._enlargeBuffer("kernel", layermem["kernel"])

            if (
                "weight_name" in op.params
                and self._isTranable(op.params["weight_name"])
                and op.params["op"] != "TRANSPOSE_CONV_2D"
            ):
                size = int(op.get_weights_size())
                self.buffers["trainable"] += size
                layermem["trainable"] = size
                layermem["weight"] = 0
            else:
                layermem["weight"] = int(op.get_weights_size())
            if "bias_name" in op.params and self._isTranable(op.params["bias_name"]):
                size = int(op.get_bias_size())
                self.buffers["trainable"] += size
                if "trainable" in layermem:
                    layermem["trainable"] += size
                else:
                    layermem["trainable"] = size
                layermem["bias"] = 0
            else:
                layermem["bias"] = int(op.get_bias_size())
            # if it is float32 op, then their wegiths/bias should from SRAM buffers
            if op.params["input_dtype"] != "int8":
                layermem["scale"] = 0
                layermem["bias"] = 0
                layermem["weight"] = 0
            self.__increaseFlash(layermem["weight"])
            self.__increaseFlash(layermem["bias"])
            self.__increaseFlash(layermem["scale"])

            self.layermem.append(layermem)

        # assign data dtype for each tensor for visualization
        # we need to figure out training_weight and training_activation here
        # for training_weight, it should contain weights of "transpose conv"
        # then, other tensors in training can be categorized as training activation
        training_start_idx = _find_training_idx(self.layer)
        # assign every tenosrs labeled as TTYPE_INFERNECE after the index as TTYPE_TRAINING_ACTIVATION
        for r in self.allocator.rectangles:
            if r["type"] == TTYPE_INFERNECE and r["end"] > training_start_idx:
                r["type"] = TTYPE_TRAINING_ACTIVATION
        # for each tranpose conv, find it
        for i, op in enumerate(self.layer):
            if op.params["op"] == "TRANSPOSE_CONV_2D":
                # if any tenosr used by this layer
                for r in self.allocator.rectangles:
                    if r["end"] <= training_start_idx:
                        continue
                    if r["name"] == op.params["weight_name"]:
                        r["type"] = TTYPE_TRAINING_WEIGHTS

        # Now that every rectangle exists, hand the allocator each tensor that is
        # can be overwritten and the output that can overwrite it.
        for op in self.layer:
            for t in op.input_tensors:
                if t.inplace:
                    self.allocator.markInplace(
                        t.allocator_idx, t.inplace_tensor_out.allocator_idx, t.gap
                    )

        # Allocating a memory location for each tensor/rectangle
        self.allocator.allocate()
        # self.allocator.visualize(self.mem_visual_path)
        self._enlargeBuffer("input_output", self.allocator.get_peak())

        # sanity check, see if all tensors have been allocated
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            for cnt, t in enumerate(op.input_tensors):
                assert t.allocator_idx is not None
            for cnt, t in enumerate(op.output_tensors):
                assert t.allocator_idx is not None

        # assign the address according to placement
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            for cnt, t in enumerate(op.input_tensors):
                if cnt == 0:
                    op.params["input_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input_buf_add"] = "front"
                elif cnt == 1:
                    op.params["input2_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input2_buf_add"] = "front"
                elif cnt == 2:
                    op.params["input3_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input3_buf_add"] = "front"
                op.input_tensors[cnt].buffer_name = "buffer0"
                op.input_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)
            for cnt, t in enumerate(op.output_tensors):
                if cnt == 0:
                    op.params["output_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["output_buf_add"] = "front"
                    op.output_tensors[cnt].buffer_name = "buffer0"
                    op.output_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)
                if cnt == 1:
                    op.params["output2_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["output2_buf_add"] = "front"
                    op.output_tensors[cnt].buffer_name = "buffer0"
                    op.output_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)

        # calculate peak mem
        self.peakmem = (
            self.allocator.get_peak() + self.buffers["im2col"] + self.buffers["kernel"]  # + self.buffers["trainable"]
        )

    def dumpLayerIndex(self):
        # header
        print("-" * 14 + " Tensor Allocation Details " + "-" * 14)
        print(" #op |   operator type   | input index | output index |")
        for cnt, l in enumerate(self.layer):
            operator_num = "#" + str(cnt)
            type = str(l.params["op"])
            input_tensor = ""
            for cnt_inp, inp in enumerate(l.input_tensors):
                input_tensor += str(inp.allocator_idx)
                if cnt_inp < len(l.input_tensors) - 1:
                    input_tensor += ","
            output_tensor = str(l.output_tensors[0].allocator_idx)
            string = (
                operator_num.ljust(5)
                + "|"
                + type.ljust(19)
                + "|"
                + input_tensor.ljust(13)
                + "|"
                + output_tensor.ljust(14)
                + "|"
            )
            print(string)

    def dumpLayerMem(self):
        # header
        print(
            "----------------------------------------------------  Schedule Details ----------------------------------------------------------------"  # noqa: E501
        )
        print(
            "----------------------|                      SRAM                      ||                     Flash                      |             |"  # noqa: E501
        )
        print(
            "----------------------|  activation  |  runtime  | trainable  |  sum   ||   weight   |   bias   |  scale   |     sum     |     MAC     |"  # noqa: E501
        )

        layermem = self.layermem
        self.__dumpMemInfo(layermem)

    def __dumpMemInfo(self, layermem):
        string = "-------Schedule-------|"
        maxActive = self.buffers["input_output"]
        maxRuntime = self.buffers["im2col"] + self.buffers["kernel"]
        maxTrainable = self.buffers["trainable"]
        totalWeight = self.__sumKey(layermem, "weight")
        totalBias = self.__sumKey(layermem, "bias")
        totalScale = self.__sumKey(layermem, "scale")
        totalMAC = self.__sumKey(layermem, "MAC")
        string += str(maxActive).ljust(14) + "|"
        string += str(maxRuntime).ljust(11) + "|"
        string += str(maxTrainable).ljust(12) + "|"
        string += str(maxActive + maxRuntime + maxTrainable).ljust(8) + "||"
        string += str(totalWeight).ljust(12) + "|"
        string += str(totalBias).ljust(10) + "|"
        string += str(totalScale).ljust(10) + "|"
        string += str(totalWeight + totalBias + totalScale).ljust(13) + "|"
        string += str(totalMAC).ljust(13) + "|"
        print(string)
        for i, _ in enumerate(layermem):
            layer_info = self.layer[i].get_layer_info()
            string = ""
            string += str(i) + ":" + layer_info["op"]
            string = string.ljust(22) + "|"
            SRAM = 0
            if "activation" in layermem[i]:
                substr = (
                    str(layermem[i]["activation"]) + " (" + "{:.0%}".format(layermem[i]["activation"] / maxActive) + ")"
                )
                string += substr.ljust(14) + "|"
                SRAM += layermem[i]["activation"]
            if "runtime" in layermem[i]:
                sbuf = layermem[i]["runtime"] + layermem[i]["kernel"]
                substr = str(sbuf) + " (" + "{:.0%}".format(sbuf / maxRuntime) + ")"
                string += substr.ljust(11) + "|"
                SRAM += sbuf
            else:
                string = string.ljust(49) + "|"
            if "trainable" in layermem[i]:
                substr = (
                    str(layermem[i]["trainable"])
                    + " ("
                    + "{:.0%}".format(layermem[i]["trainable"] / maxTrainable)
                    + ")"
                )
                string += substr.ljust(12) + "|"
                SRAM += layermem[i]["trainable"]
            else:
                string = string.ljust(62) + "|"

            # SRAM end
            string += str(SRAM)
            string = string.ljust(71) + "||"
            flash = 0
            if "weight" in layermem[i]:
                substr = (
                    str(layermem[i]["weight"])
                    + " ("
                    + "{:.0%}".format(layermem[i]["weight"] / (totalWeight + 0.0001))
                    + ")"
                )
                string += str(substr).ljust(12) + "|"
                flash += layermem[i]["weight"]
            if "bias" in layermem[i]:
                substr = (
                    str(layermem[i]["bias"]) + " (" + "{:.0%}".format(layermem[i]["bias"] / (totalBias + 0.0001)) + ")"
                )
                string += str(substr).ljust(10) + "|"
                flash += layermem[i]["bias"]
            if "scale" in layermem[i]:
                substr = (
                    str(layermem[i]["scale"]) + " (" + "{:.0%}".format(layermem[i]["scale"] / totalScale + 0.0001) + ")"
                )
                string += str(substr).ljust(10) + "|"
                flash += layermem[i]["scale"]

                if flash > 0:
                    string += (
                        str(flash)
                        + " ("
                        + "{:.0%}".format(flash / (totalWeight + totalBias + totalScale + 0.0001))
                        + ")"
                    )
                    string = string.ljust(121) + "|"
            # flash end
            if "MAC" in layermem[i]:
                substr = str(layermem[i]["MAC"]) + " (" + "{:.0%}".format(layermem[i]["MAC"] / totalMAC) + ")"
                string += str(substr).ljust(13) + "|"
            print(string)

    def __sumKey(self, layers, key):
        result = 0
        for _, layer in enumerate(layers):
            if key in layer:
                result += layer[key]

        return result

    def getBuffers(self):
        return self.buffers

    # Maximum binary size: This should be updated if any change in the inference side
    # TODO: Combine with code generation to get more accurate result
    def profileResult(self):
        return self.peakmem, self.flash + self.bias + self.scale + int(self.code * 1024)

    def __increaseFlash(self, size):
        self.flash += int(size)

    def _enlargeBuffer(self, buf_str, size):
        if buf_str == "input_output" or buf_str == "residual":
            self.buffers[buf_str] = max(self.buffers[buf_str], int(size))
        else:
            if buf_str not in self.buffers:
                self.buffers[buf_str] = size
            else:
                self.buffers[buf_str] = max(self.buffers[buf_str], size)


def _find_training_idx(layers):
    idx = len(layers)
    for cnt, l in enumerate(layers):
        if l.params["op"] in ["CAST"]:
            return cnt
    return idx
