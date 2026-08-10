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
#include "genModelShape.h"
#include "genNN.h"
#include "pico/stdlib.h"

// Define a magic anchor pattern to fill the unallocated stack space
#define STACK_MAGIC_PATTERN 0xDEADBEEF
#define AVERAGE_RUN         1

// Pull in the linker symbols tracking where the stack boundaries sit
extern uint32_t __StackLimit;

uint32_t* start_stack_top; 
void paint_stack(void) {
    // Get the current local address of the stack pointer right now
    volatile uint32_t* sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    start_stack_top = (uint32_t*)sp;

    // Fill everything between the current pointer and the hard bottom limit
    uint32_t* limit = &__StackLimit;
    while (limit < sp) {
        *limit = STACK_MAGIC_PATTERN;
        limit++;
    }
}

uint32_t get_used_stack_bytes(void) {
    uint32_t* limit = &__StackLimit;
    // The stack grows down, so start_stack_top is the higher address
    uint32_t total_stack_size = (uint32_t)(start_stack_top - &__StackLimit) * sizeof(uint32_t);
    uint32_t unused_words = 0;

    // Count how many magic patterns are still untouched
    while (*limit == STACK_MAGIC_PATTERN) {
        unused_words++;
        limit++;
    }
    return total_stack_size - unused_words * sizeof(uint32_t);
}

void init_cycle_counter() {
    // 1. Enable the trace and debug blocks
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    // 2. Clear the current cycle counter register
    DWT->CYCCNT = 0;

    // 3. Enable the cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

typedef void(*invoke_fun)(void*);
uint32_t measure_code_cycles(invoke_fun fun) {
    uint32_t start = DWT->CYCCNT;
    fun(NULL);
    return (DWT->CYCCNT - start);
}

uint64_t measure_code_duration(invoke_fun fun) {
    // Snap the starting cycle count
    uint64_t start_time = time_us_64();
    for (int i=0; i < AVERAGE_RUN; i++) {
        fun(NULL);
    }
    return (time_us_64() - start_time) / AVERAGE_RUN;
}

void load_input(int8_t* input){
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (int8_t)(i % 256 - 128);
    }
}

int32_t get_checksum(const int8_t* output) {
    int32_t total = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        total += output[i];
    }
    return total;
}

int main(void) {
    init_cycle_counter();
    stdio_init_all();

    // Call this at the very top of main before doing heavy processing
    paint_stack();
    while (!stdio_usb_connected()) {
        sleep_ms(10);
    }
    int8_t* input = (int8_t*)getInput();
    int8_t* output = (int8_t*)getOutput();

    while(true) {
        load_input(input);
        uint32_t cycle_count = measure_code_cycles(invoke_inf);
        uint32_t stack_size = get_used_stack_bytes();
        int32_t checksum = get_checksum(output);
        load_input(input);
        uint64_t duration_us = measure_code_duration(invoke_inf);

        /* Two runs of the same work: the checksums must agree, otherwise the
        * second inference saw different input from the first. */
        printf("cycles       = %lu\n", (unsigned long)cycle_count);
        printf("duration     = %llu us\n", (unsigned long long)duration_us);
        printf("stack used = %lu bytes\n", (unsigned long)stack_size);
        printf("checksum     = %ld\n", (long)checksum);
    }

    return 0;
}
