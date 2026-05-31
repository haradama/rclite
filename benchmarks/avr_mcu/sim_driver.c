/* libsimavr cycle-count driver for the AVR rc_predict benchmark.
 *
 * Loads an ATmega328P ELF and runs it under simavr (cycle-accurate, fully
 * deterministic). The firmware (main_bench.c) signals via the general-purpose
 * I/O registers, captured here at the exact write instant:
 *
 *   GPIOR0 (data 0x3E):  1 = start timed region (capture avr->cycle)
 *                        2 = end timed region   (capture avr->cycle)
 *                        3 = all done           (stop the simulation)
 *   GPIOR1 (data 0x4A):  1 = PARITY_OK, 0 = PARITY_FAIL
 *
 * Prints `avr_cycles: <delta>` and `parity: OK|FAIL`. The cycle delta is the
 * exact AVR cycle count of rc_predict over the timed window (plus two `out`
 * marker instructions, negligible). Build:
 *   gcc sim_driver.c -lsimavr -o sim_driver
 */
#include <simavr/sim_avr.h>
#include <simavr/sim_elf.h>
#include <simavr/sim_io.h>
#include <stdio.h>

static avr_cycle_count_t c_start = 0, c_end = 0;
static int finished = 0, parity = -1;

static void on_gpior0(struct avr_t *avr, avr_io_addr_t addr, uint8_t v, void *p) {
    if (v == 1) c_start = avr->cycle;
    else if (v == 2) c_end = avr->cycle;
    else if (v == 3) finished = 1;
}

static void on_gpior1(struct avr_t *avr, avr_io_addr_t addr, uint8_t v, void *p) {
    parity = v;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s firmware.elf\n", argv[0]);
        return 2;
    }
    elf_firmware_t f = {{0}};
    if (elf_read_firmware(argv[1], &f)) {
        fprintf(stderr, "elf read failed: %s\n", argv[1]);
        return 2;
    }
    avr_t *avr = avr_make_mcu_by_name("atmega328p");
    if (!avr) {
        fprintf(stderr, "unknown mcu atmega328p\n");
        return 2;
    }
    avr_init(avr);
    avr->frequency = 16000000;
    avr_load_firmware(avr, &f);
    avr_register_io_write(avr, 0x3E, on_gpior0, NULL);   /* GPIOR0 */
    avr_register_io_write(avr, 0x4A, on_gpior1, NULL);   /* GPIOR1 */

    int state = cpu_Running;
    long guard = 0;
    while (!finished && state != cpu_Done && state != cpu_Crashed
           && guard++ < 2000000000L) {
        state = avr_run(avr);
    }
    if (!finished) {
        fprintf(stderr, "firmware did not signal done (state=%d)\n", state);
        return 3;
    }
    printf("avr_cycles: %llu\n", (unsigned long long)(c_end - c_start));
    printf("parity: %s\n", parity == 1 ? "OK" : "FAIL");
    return 0;
}
