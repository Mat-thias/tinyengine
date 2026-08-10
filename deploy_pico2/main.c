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

#include "genNN.h"
#include "genModelShape.h"
#include "pico/stdlib.h"

int main(void) {
    stdio_init_all();

    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }

    int8_t* input = (int8_t*)getInput();
    int8_t* output = (int8_t*)getOutput();

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
