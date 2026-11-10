from .basic_utils import basicOperator, deep_copy_dicts, op_inplace_type

__all__ = [
    "SingleInputPlaceHolderOp",
    "DoubleInputPlaceHolderOp",
    "NInputPlaceHolderOp",
]

# What every placeholder has: the op it stands for and its one output.
default_params = {
    "op": None,
    "output_idx": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # Nothing is known about a layer with no kernel, so it is never assumed to overwrite what it
    # reads: it always gets an output buffer of its own, as in dual buffer mode.
    "inplace": op_inplace_type.force_not_inplace,
    "op_needs_right_shift": False,
    "color": "#d9d9d9",  # grey, unless the parser picks one
}


class _PlaceHolderOp(basicOperator):
    """An op with no operator class of its own, scheduled from the tensors it reads and writes.

    Its lifetimes and sizes are worked out like any other layer's, and it is drawn in the
    lifecycle plot. No kernel for it exists in this tree, so its output never shares memory with
    its inputs; a parser may pick its colour but not how it is placed.

    The variants differ only in how many tensors they read, which decides the group the
    scheduler places them in.
    """

    def __init__(self, params: dict) -> None:
        assert not {"inplace", "op_needs_right_shift"} & params.keys(), (
            f"{params.get('op')}: a placeholder is always placed apart from its inputs, so its parser "
            "cannot set inplace or op_needs_right_shift"
        )
        self.params = deep_copy_dicts(default_params)
        self.params.update(self._input_default_params())
        # every key the parser gives is kept, not only the ones declared above
        self.params.update(params)
        super().__init__()
        self._add_inputs()
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

    def _input_default_params(self) -> dict:
        """The input fields this variant reads."""
        raise NotImplementedError

    def _add_inputs(self) -> None:
        """One input tensor per input this variant reads."""
        raise NotImplementedError


class SingleInputPlaceHolderOp(_PlaceHolderOp):
    """A placeholder reading one tensor."""

    def _input_default_params(self) -> dict:
        return {"input_idx": None, "input_dim": None, "input_h": None, "input_w": None, "input_c": None}

    def _add_inputs(self) -> None:
        p = self.params
        self._add_input(p["input_idx"], p["input_dtype"], p["input_c"], p["input_w"], p["input_h"])


class DoubleInputPlaceHolderOp(_PlaceHolderOp):
    """A placeholder reading two tensors."""

    def _input_default_params(self) -> dict:
        return {
            "input_idx": None, "input_dim": None, "input_h": None, "input_w": None, "input_c": None,
            "input2_idx": None, "input2_dim": None, "input2_h": None, "input2_w": None, "input2_c": None,
            "input2_dtype": "int8",
        }

    def _add_inputs(self) -> None:
        p = self.params
        self._add_input(p["input_idx"], p["input_dtype"], p["input_c"], p["input_w"], p["input_h"])
        self._add_input(p["input2_idx"], p["input2_dtype"], p["input2_c"], p["input2_w"], p["input2_h"])


class NInputPlaceHolderOp(_PlaceHolderOp):
    """A placeholder reading any number of tensors, given as one entry each in `inputs`.

    Each entry holds the tensor's idx, h, w and c, and optionally its dtype.
    """

    def _input_default_params(self) -> dict:
        return {"inputs": None}

    def _add_inputs(self) -> None:
        assert self.params["inputs"], f"{self.params['op']} reads no `inputs`"
        for t in self.params["inputs"]:
            self._add_input(t["idx"], t.get("dtype", self.params["input_dtype"]), t["c"], t["w"], t["h"])
