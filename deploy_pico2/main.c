/* ----------------------------------------------------------------------
 * Project: Right In-Place Convolution
 * Title:   main.c
 *
 * Reference papers:
 *    Yet to be published
 * Contact authors:
 *  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
 *  - Tolulope Matthew Busoye, matthew@powerlabstech.com
 *  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
 *
 * Target ISA:  ARMv8-M (Cortex-M33, RP2350)
 * -------------------------------------------------------------------- */

/* Run the generated model once on a deterministic int8 image and print the
 * first outputs over USB CDC, for comparing against the host build. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "RP2350.h"  // Includes the CMSIS register definitions
#include "genNN.h"
#include "genModelShape.h"
#include "pico/stdlib.h"

#define INPUT_SIZE (160 * 160 * 3)
#define NUM_CLASSES 1000

// Define a magic anchor pattern to fill the unallocated stack space
#define STACK_MAGIC_PATTERN 0xDEADBEEF

// Pull in the linker symbols tracking where the stack boundaries sit
extern uint32_t __StackLimit;

void paint_stack(void) {
    // Get the current local address of the stack pointer right now
    volatile uint32_t* sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));

    // Fill everything between the current pointer and the hard bottom limit
    uint32_t* limit = &__StackLimit;
    while (limit < sp) {
        *limit = STACK_MAGIC_PATTERN;
        limit++;
    }
}

uint32_t get_unused_stack_bytes(void) {
    uint32_t* limit = &__StackLimit;
    uint32_t unused_words = 0;

    // Count how many magic patterns are still untouched
    while (*limit == STACK_MAGIC_PATTERN) {
        unused_words++;
        limit++;
    }
    return unused_words * sizeof(uint32_t);
}

void init_cycle_counter() {
    // 1. Enable the trace and debug blocks
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    // 2. Clear the current cycle counter register
    DWT->CYCCNT = 0;

    // 3. Enable the cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t measure_code_cycles() {
    // Snap the starting cycle count
    uint32_t start = DWT->CYCCNT;

    // ----------------------------------------
    // INSERT YOUR PROFILING TARGET HERE
    // ----------------------------------------
    __asm volatile("nop");
    __asm volatile("nop");
    __asm volatile("nop");
    __asm volatile("nop");
    __asm volatile("nop");
    for (volatile int i = 0; i < 1000; i++) {
        __asm volatile("nop");
        __asm volatile("nop");
    }
    // ----------------------------------------

    // Snap the ending cycle count
    uint32_t end = DWT->CYCCNT;

    // Direct subtraction automatically accounts for 32-bit overflows
    return (end - start);
}

int main(void) {
    stdio_init_all();
    // Get raw microseconds since chip boot
    uint64_t start_time = time_us_64();
    uint64_t end_time = time_us_64();
    uint32_t array[74];
    printf("duration = %llu microseconds\n", end_time - start_time);
    init_cycle_counter();
    // for (int i = 0; i < 74; i++) {
    //     array[i] = i;
    // }

    // for (int i = 0; i < 74; i++) {
    //     printf("array[%d] = %d\n", i, array[i]);
    // }
    while (!stdio_usb_connected()) {
        sleep_ms(10);
    }
    start_time = time_us_64() - start_time;
    sleep_ms(5);
    sleep_ms(1000);

    for (int i = 0; i < 10; i++) {
        printf("duration[%d] = %llu microseconds\n", i, time_us_64() - start_time);
    }
    printf("USB connected after %llu microseconds\n", start_time);

    while (true) {
        __asm volatile("nop");
        __asm volatile("nop");
        uint32_t elapsed = measure_code_cycles();
        printf("Elapsed CPU cycles: %lu\n", elapsed);
        sleep_ms(1000);
    }

    // Call this at the very top of main before doing heavy processing
    paint_stack();

    // while (true) {
    //     // Run your application functions here...

    //     // Print the remaining buffer space available before a crash occurs
    //     printf("Unused stack space safety margin: %u bytes\n", get_unused_stack_bytes());
    //     sleep_ms(2000);
    // }

    volatile int8_t* input = (int8_t*)getInput();
    volatile int8_t* output = (int8_t*)getOutput();

    srand(42);
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (int8_t)(i % 256 - 128);
    }

    invoke(NULL);

    for (int i = 0; i < 10; i++) {
        printf("output[%d] = %d\n", i, output[i]);
    }

    return 0;
}
