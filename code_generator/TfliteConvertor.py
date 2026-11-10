# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   TfliteConvertor.py
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

import logging

import code_generator.converters.tflite_parser as TF_Parser
from code_generator.converters.tflite_parser.mean1dto2d import MEAN2D
from code_generator.converters.tflite_parser.utils import (
    get_input_tensors, get_nhwc_from_shape, get_np_from_wrapper, get_output_tensors, getOpCodeStr, is_constant
)

from .constant import SKIP_OPs
from .tflite import Model


# Parse tflite model into TinyEngine IR format
class TfliteConvertor(object):
    def __init__(self, filepath):
        # path to the tflite file
        self.filepath = filepath
        self.model = self.loadTFmodel(filepath)
        self.subgraph = self.model.Subgraphs(0)
        self.layer = []
        # every PAD folded into the op that reads it, by the tensor the PAD produces
        self.tmpPADIndice = dict()
        # every RESHAPE folded away, by the tensor it produces; whatever reads it reads its input
        self.folded_reshapes = dict()
        self.skip_transpose = None
        self.average_1D_to_2D_holder = MEAN2D()  # For merging 1D to 2D

    # public functions
    def loadTFmodel(self, filepath):
        buf = open(filepath, "rb").read()
        return Model.Model.GetRootAsModel(buf, 0)

    def dumpModelInfo(self):
        version = self.model.Version()
        print("Model version:", version)
        description = self.model.Description().decode("utf-8")
        print("Description:", description)
        subgraph_len = self.model.SubgraphsLength()
        print("Subgraph length:", subgraph_len)

        self.dumpLayerInfo()

    def dumpLayerInfo(self):
        print("Layer length:", len(self.layer))

        # print brief info about each layer
        for i, layer in enumerate(self.layer):
            if self.layer[i]["op"] == "ADD":
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    ",input2_idx:",
                    layer["input2_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )
            else:
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )

    def parseOperatorInfo(self):
        operators_len = self.subgraph.OperatorsLength()

        skip_next_ops = 0
        for i in range(operators_len):
            if skip_next_ops > 0:
                skip_next_ops -= 1
                continue

            op = self.subgraph.Operators(i)

            # A flatten that keeps the batch dynamic is SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE. The first
            # three only work out the new shape, so they are skipped and the RESHAPE is folded away.
            if getOpCodeStr(op, self.model) == "SHAPE":
                chain = [getOpCodeStr(self.subgraph.Operators(j), self.model) for j in range(i, min(i + 4, operators_len))]
                assert chain == ["SHAPE", "STRIDED_SLICE", "PACK", "RESHAPE"], (
                    f"op {i}: SHAPE starts {chain}, expected the flatten SHAPE, STRIDED_SLICE, PACK, RESHAPE"
                )
                skip_next_ops = 2
                continue

            if i < operators_len - 2:
                next_op = self.subgraph.Operators(i + 1)
                next_next_op = self.subgraph.Operators(i + 2)
                three_op_sequence = [op, next_op, next_next_op]

                op_str = getOpCodeStr(op, self.model)
                if  op_str == "MEAN":
                    prev_op = self.subgraph.Operators(i - 1) if i > 0 else None
                    prev_op_str = getOpCodeStr(prev_op, self.model) 
                    next_op_str = getOpCodeStr(next_op, self.model) 
                    next_next_op_str = getOpCodeStr(next_next_op, self.model)

                    # A MEAN over H and W is a whole global average pool on its own. Any other
                    # MEAN must come as a pair: parse_mead1dto2d holds the first and emits
                    # nothing, the second builds the avgpool.
                    if prev_op_str != "MEAN" and TF_Parser.mean_axes(op, self.model) != {1, 2}:
                        assert next_op_str == "MEAN", (
                            f"op {i} MEAN is followed by {next_op_str}, expected MEAN; a lone MEAN "
                            "emits no layer and its held shape would pair with the next one in the graph"
                        )
                        assert next_next_op_str != "MEAN", (
                            f"ops {i}..{i + 2} are three consecutive MEANs, expected a pair; "
                            "the third would open a new pair and match against a later MEAN"
                        )
                elif op_str == "TRANSPOSE":
                    next_op_str = getOpCodeStr(next_op, self.model)
                    next_next_op_str = getOpCodeStr(next_next_op, self.model)
                    # The transpose is skipped, because we currently on support on that is followed by a global
                    # average pooling and an fc. The global average pool reduces the HxW to 1x1, so it is 
                    # comfortably skipped
                    assert next_op_str == "MEAN" and next_next_op_str == "MEAN", (
                        f"op {i} TRANSPOSE is followed by {next_op_str},{next_next_op_str}, "
                        "expected MEAN,MEAN; otherwise it is dropped and the data never permuted"
                    )

                if self.checkIfRequireSEelementmult(three_op_sequence):
                    logging.info("found SE block")
                    skip_next_ops = 2
                    #         -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
                    #  DWCONV                                        |            -> MUL |
                    #                                                |   SEelementmult   |
                    SEelementmult_op = TF_Parser.parse_SEelement(three_op_sequence, self.model, self.layer)

                    self.layer.append(SEelementmult_op)
                    continue

            # parse the op
            self._handleOperator(op)

    # handle one op and parse it into layers[] for supported operators
    def _handleOperator(self, op):
        op_code_str = getOpCodeStr(op, self.model)
        layers_before = len(self.layer)
        if op_code_str == "CONV_2D":
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
        elif op_code_str == "ADD" and not any(is_constant(t) for t in get_input_tensors(op, self.model)):
            # two activations of one shape, which main's Add places in place
            self.layer.append(TF_Parser.parse_add(op, self.model))
        elif op_code_str in ("ADD", "SUB", "MUL"):
            self.layer.append(TF_Parser.parse_elementwise(op, self.model))
        elif op_code_str == "AVERAGE_POOL_2D":
            self.layer.append(TF_Parser.parse_avgpool(op, self.model, self.tmpPADIndice))
        elif op_code_str == "DEPTHWISE_CONV_2D":
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
        elif op_code_str == "PAD":
            self._convert_PAD(op)
        elif op_code_str == "RESIZE_NEAREST_NEIGHBOR":
            self.layer.append(TF_Parser.parse_upsample(op, self.model))
        elif op_code_str == "MAX_POOL_2D":
            self.layer.append(TF_Parser.parse_maxpool(op, self.model, self.tmpPADIndice))
        elif op_code_str == "MEAN" and TF_Parser.mean_axes(op, self.model) == {1, 2}:
            self.layer.append(TF_Parser.parse_mean(op, self.model))
        elif op_code_str == "MEAN":
            ret_op = TF_Parser.parse_mead1dto2d(op, self.model, self.average_1D_to_2D_holder)
            if ret_op is not None:
                # TODO: This only handle a specific graph: TRANSPOSE -> MEAN -> MEANS
                if self.skip_transpose is not None:
                    ret_op.params["input_idx"] = self.skip_transpose.input_idx
                    ret_op.input_tensors[0].graph_idx = str(self.skip_transpose.input_idx)
                self.layer.append(ret_op)
        elif op_code_str == "TRANSPOSE":
            self._convert_TRANSPOSE(op)
        elif op_code_str == "FULLY_CONNECTED":
            self.layer.append(TF_Parser.parse_fc(op, self.model))
        elif op_code_str == "RESHAPE":
            self._convert_RESHAPE(op)
        elif op_code_str == "SOFTMAX":
            # There is no SOFTMAX layer, so it may only end the model, which then returns the logits it
            # reads: one value per class, as large as the SOFTMAX's own output.
            output_idx = get_output_tensors(op, self.model)[0].tensor_idx
            assert output_idx in self.subgraph.OutputsAsNumpy(), (
                f"SOFTMAX writing tensor {output_idx} is not the model's output, so it cannot be dropped"
            )
        elif op_code_str in SKIP_OPs:
            pass
        else:
            raise NotImplementedError(f"Unsupported {op_code_str}")

        # a layer reading a folded RESHAPE reads what the RESHAPE read
        for layer in self.layer[layers_before:]:
            self._read_through_folded_reshapes(layer)

    def _read_through_folded_reshapes(self, layer):
        for t in layer.input_tensors:
            fold = self.folded_reshapes.get(t.graph_idx)
            if fold is not None:
                t.graph_idx = str(fold.input_idx)
        for key in ("input_idx", "input2_idx"):
            fold = self.folded_reshapes.get(str(layer.params.get(key)))
            if fold is not None:
                layer.params[key] = fold.input_idx

    def _consumers(self, tensor_idx):
        """The ops that read tensor_idx."""
        ops = (self.subgraph.Operators(i) for i in range(self.subgraph.OperatorsLength()))
        return [op for op in ops if tensor_idx in op.InputsAsNumpy()]

    #         -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
    #  DWCONV                                        |            -> MUL |
    #                                                |    Fuse Target    |
    def checkIfRequireSEelementmult(self, three_op_sequence):
        if (
            getOpCodeStr(three_op_sequence[0], self.model) == "ADD"
            and getOpCodeStr(three_op_sequence[1], self.model) == "MUL"
            and getOpCodeStr(three_op_sequence[2], self.model) == "MUL"
        ):
            return True
        return False

    def _convert_PAD(self, op):
        # get input, weight, and output tensors
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # [[batch], [top, bottom], [left, right], [channel]]: only H and W fold into a conv or pool
        paddings = get_np_from_wrapper(input_tensors[1])
        assert not paddings[0].any() and not paddings[3].any(), (
            f"PAD {paddings.tolist()} pads the batch or channel axis, which cannot fold into a conv or pool"
        )

        # fuse pad into the conv or pool that reads it
        self.tmpPADIndice[output_tensor.tensor_idx] = PAD_tensorIndice(
            input_tensor.tensor_idx, output_tensor.tensor_idx,
            pad_t=int(paddings[1][0]), pad_b=int(paddings[1][1]),
            pad_l=int(paddings[2][0]), pad_r=int(paddings[2][1]),
        )

    def _convert_TRANSPOSE(self, op):
        # get input, weight, and output tensors
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
        _, output_c, output_h, output_w = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
        # The only transpose supported is one that converts NHWC to NCHW, which mostly occurs just before a
        # global average pooling and fully connected layer
        assert_reason = f"transpose {(input_h, input_w, input_c)} -> {(input_h, input_w, input_c)} is not an NHWC -> NCHW relabel"
        assert input_h == output_h, assert_reason
        assert input_w == output_w, assert_reason
        assert input_c == output_c, assert_reason

        # fuse pad into conv
        self.skip_transpose = SkipTensorIndices(input_tensor.tensor_idx, output_tensor.tensor_idx)

    def _convert_RESHAPE(self, op):
        input_tensor = get_input_tensors(op, self.model)[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # Folding the RESHAPE away is only sound where it moves no byte for whoever reads it: a
        # FULLY_CONNECTED or a SOFTMAX reads the values in order, and a conv over a 1x1 input reads its
        # channels as they lie. A RESHAPE nothing reads is the model's output, and goes as well.
        for consumer in self._consumers(output_tensor.tensor_idx):
            kind = getOpCodeStr(consumer, self.model)
            if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                _, h, w, _ = get_nhwc_from_shape(get_input_tensors(consumer, self.model)[0].tensor.ShapeAsNumpy())
                assert h == w == 1, f"RESHAPE read by a {kind} over a {h}x{w} input, which it would reorder"
            else:
                assert kind in ("FULLY_CONNECTED", "SOFTMAX"), f"RESHAPE read by {kind}, which cannot read through it"

        self.folded_reshapes[str(output_tensor.tensor_idx)] = SkipTensorIndices(
            input_tensor.tensor_idx, output_tensor.tensor_idx
        )


class SkipTensorIndices(object):
    """An op folded away: what it reads, and what it would have produced."""

    def __init__(self, input_idx, output_idx):
        self.input_idx = input_idx
        self.output_idx = output_idx


class PAD_tensorIndice(SkipTensorIndices):
    """A PAD folded into the op that reads it, with the padding it adds to each side."""

    def __init__(self, input_idx, output_idx, pad_t, pad_b, pad_l, pad_r):
        super().__init__(input_idx, output_idx)
        self.pad_t = pad_t
        self.pad_b = pad_b
        self.pad_l = pad_l
        self.pad_r = pad_r
