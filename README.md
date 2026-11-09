# TinyEngine memory scheduler for right in-place convolution

A reduced fork of [TinyEngine](https://github.com/mit-han-lab/tinyengine) (MIT HAN Lab), kept only to schedule the
activation memory of int8 TFLite models on microcontrollers. It parses a model, works out every tensor's lifetime, and
places the tensors in memory, letting a layer's output overwrite its input where the kernel allows it.

It does not generate C or run a model: upstream's code generator, kernels, tutorials, demos and training engine are
removed.

| Part | Where |
| --- | --- |
| TFLite parser | `code_generator/TfliteConvertor.py`, `code_generator/converters/tflite_parser/` |
| Tensor lifetimes and in-place candidates | `code_generator/GeneralMemoryScheduler.py` |
| Allocators: MILP (right in-place) and first fit | `code_generator/allocator/` |
| Patch-based planning (MCUNetV2) | `code_generator/InputResizer.py`, `code_generator/PatchBasedUtil.py` |

## Setup

The tree is checked out inside the `right_inplace_conv` repository and run from its own root. The scheduler imports
the in-place gap formula from `../right_inplace.py`, and the examples load models through the `mcunet` package next to
it.

```bash
pip install -r requirements.txt
```

The MILP allocator solves with [pyomo](https://www.pyomo.org/) and SCIP by default. Put `scip` on `PATH`, or name the
directory it is in:

```bash
export TINYENGINE_MILP_SOLVER_PATH=/path/to/solvers   # TINYENGINE_MILP_SOLVER picks another solver
```

## Usage

Schedule a model from the MCUNet zoo and print its peak activation memory:

```bash
python examples/mcunet_inx.py -x 0 -a MILP -m full     # mcunet-in0; -n takes any other zoo id
```

`-a` is `MILP` or `first_fit`, `-m` is the in-place mode (`full`, `depthwise_only` or `dual_buffer`), and `-f` is where
the tensor lifecycle plot goes (`./lifecycle.png` by default).

Schedule the VWW model patch by patch:

```bash
PYTHONPATH=. python examples/vww_patchbased.py -p 2 -s 5   # a 2x2 patch grid ending before layer 5
```

## License

MIT, as upstream; see [LICENSE](LICENSE). The scheduler, the parser and patch-based planning build on MCUNet and
MCUNetV2:

```
@article{lin2020mcunet,
  title={Mcunet: Tiny deep learning on iot devices},
  author={Lin, Ji and Chen, Wei-Ming and Lin, Yujun and Gan, Chuang and Han, Song},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{lin2021mcunetv2,
  title={MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning},
  author={Lin, Ji and Chen, Wei-Ming and Cai, Han and Gan, Chuang and Han, Song},
  booktitle={Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
