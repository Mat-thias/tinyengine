/* Run the generated model once on a random int8 image. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "genNN.h"
#include "pico/stdlib.h"

#define INPUT_SIZE (160 * 160 * 3)
#define NUM_CLASSES 1000

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
