/* ----------------------------------------------------------------------
 * Project: Right In-Place Convolution
 * Title:   right_inplace.h
 *
 * Reference papers:
 *    Yet to be published
 * Contact authors:
 *  - Opegbemi Matthias Busoye, busoyeopegbemimatthias@gmail.com
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#ifndef TINYENGINE_INCLUDE_RIGHT_INPLACE_H_
#define TINYENGINE_INCLUDE_RIGHT_INPLACE_H_

#include <stdint.h>
#include <string.h>

/* Include after a header that defines q7_t (tinyengine_function.h). */

/* ----------------------------------------------------------------------
 * Right-inplace prologue, shared by every kernel that can alias its output
 * onto its input.
 *
 * Such a kernel is handed a single buffer holding the input at its base, and
 * must leave the output at that same base. Before computing, the input is
 * shifted up by `gap` bytes so the output write frontier can never overtake
 * the input read frontier. `gap` is derived per layer by
 * right_inplace_gap_unpadded_ext() in code_generator/operators/right_inplace.py
 * and emitted as the last argument of the generated call.
 *
 * Rewrites `input` to point at the shifted data. It must do nothing at all
 * when the kernel is not aliasing: with output != input the caller's data is
 * already where the kernel should read it, so advancing `input` there would
 * read `gap` bytes past the start of the tensor.
 * -------------------------------------------------------------------- */
#define RIGHT_INPLACE_SHIFT_INPUT(input, output, input_x, input_y, input_ch, gap) \
	do {                                                                          \
		if ((output) == (input) && (gap) > 0) {                                   \
			q7_t *_ri_base = (input);                                             \
			uint32_t _ri_size = (uint32_t)(input_x) * (input_y) * (input_ch);     \
			(input) = (input) + (gap);                                            \
			memmove((void *)(input), (void *)_ri_base, _ri_size);                 \
		}                                                                         \
	} while (0)

#endif /* TINYENGINE_INCLUDE_RIGHT_INPLACE_H_ */
