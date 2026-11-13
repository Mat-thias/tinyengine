/* ----------------------------------------------------------------------
 * Project: Right In-Place Convolution
 * Title:   unaligned_word_access.h
 *
 * Reference papers:
 *    Yet to be published
 * Contact authors:
 *  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
 *  - Tolulope Matthew Busoye, matthew@powerlabstech.com
 *  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
 *
 * Target ISA:  ARMv6-M (Cortex-M0+), ARMv7E-M, ARMv8-M
 * -------------------------------------------------------------------- */

/* A 32-bit type that may sit at any byte address.
 *
 * Several kernels view a q15_t buffer through a q31_t pointer to process two
 * halfwords at once. Where the q15_t pointer has been advanced by an odd
 * number of elements the resulting word pointer is misaligned by two bytes:
 * in mat_mult_kernel3_input3_s8_s16_fpreq the second kernel row starts at
 * ksrc + 27, and in convolve_s8_kernel3_inputch3_stride2_pad1_fpreq the
 * padded-row path writes one q15_t before switching to word stores. Both are
 * misaligned every time, not occasionally.
 *
 * ARMv7E-M and ARMv8-M take unaligned word accesses in hardware, so this went
 * unnoticed. ARMv6-M has no such support: LDR or STR on an address that is
 * not a multiple of four raises a HardFault, which on a Cortex-M0+ escalates
 * to priority -1 and masks the interrupts the USB stack needs, so the board
 * goes silent rather than reporting anything.
 *
 * Declaring those pointers as te_q31_unaligned_t* rather than q31_t* tells
 * the compiler the address is only byte-aligned, so it emits four byte loads
 * instead of a faulting LDR. Only the pointer declarations change at the use
 * sites; the surrounding code is untouched.
 *
 * The reduced alignment applies only where the hardware needs it. Where
 * __ARM_FEATURE_UNALIGNED is set the type stays a plain int32_t, so those
 * targets compile byte-for-byte as they did before this header existed and
 * no measurement taken on them has to be repeated. That is the same macro
 * CMSIS keys write_q15x2_ia off, for the same reason.
 */

#ifndef TINYENGINE_UNALIGNED_WORD_ACCESS_H_
#define TINYENGINE_UNALIGNED_WORD_ACCESS_H_

#include <stdint.h>

#if !defined(__ARM_FEATURE_UNALIGNED) && (defined(__GNUC__) || defined(__clang__))
/* Strict-alignment target: spell out that the address may be unaligned. */
typedef int32_t te_q31_unaligned_t __attribute__((aligned(1)));
#else
/* Unaligned word access is free here, so keep the ordinary type and leave
 * code generation exactly as it was. */
typedef int32_t te_q31_unaligned_t;
#endif

#endif /* TINYENGINE_UNALIGNED_WORD_ACCESS_H_ */
