/* Wokwi-friendly Pico variant of main_template.c.
 *
 * Wokwi's wokwi-pi-pico simulator captures USB-CDC or wired-UART output
 * to its `$serialMonitor` virtual part, NOT ARM semihosting. So instead
 * of BKPT #0xAB we initialize UART0 (GP0=TX) at 115200 baud and stream
 * the same comparison output there. `EMULATOR_EXIT` is preserved as the
 * end-of-run marker for `wokwi-cli --expect-text`.
 *
 * Clock setup: enable XOSC (12 MHz crystal) and route clk_ref/clk_sys to
 * it. After cold boot only ROSC (~6 MHz, ±50% spec spread) is running,
 * so we'd see wrong baud without this. XOSC is the most predictable
 * source short of bringing up PLL_SYS.
 *
 * Placeholders (filled by examples/build_microbit*.py-style drivers):
 *   @@T_LEN@@      — number of inference steps
 *   @@X_VALUES@@   — comma-separated input samples
 *   @@Y_VALUES@@   — comma-separated host-reference predictions
 */
#include <stdint.h>
#include "rc_predict.h"

#define T_LEN @@T_LEN@@

static const float X_in[T_LEN]       = { @@X_VALUES@@ };
static const float Y_reference[T_LEN] = { @@Y_VALUES@@ };

#define REG(a) (*(volatile uint32_t *)(a))

#define RESETS_BASE   0x4000c000u
#define RESETS_DONE   (RESETS_BASE + 0x8)
#define RESETS_CLR    (RESETS_BASE + 0x3000)
#define IO_BANK0_BASE 0x40014000u
#define UART0_BASE    0x40034000u
#define CLOCKS_BASE   0x40008000u
#define XOSC_BASE     0x40024000u

#define RESET_IO_BANK0   (1u <<  5)
#define RESET_PADS_BANK0 (1u <<  8)
#define RESET_UART0      (1u << 22)

#define UART0_DR    (UART0_BASE + 0x00)
#define UART0_FR    (UART0_BASE + 0x18)
#define UART0_IBRD  (UART0_BASE + 0x24)
#define UART0_FBRD  (UART0_BASE + 0x28)
#define UART0_LCR_H (UART0_BASE + 0x2C)
#define UART0_CR    (UART0_BASE + 0x30)

#define UART_FR_TXFF       (1u <<  5)
#define UART_LCR_H_WLEN_8  (3u <<  5)
#define UART_LCR_H_FEN     (1u <<  4)
#define UART_CR_UARTEN     (1u <<  0)
#define UART_CR_TXE        (1u <<  8)

#define XOSC_CTRL    (XOSC_BASE + 0x00)
#define XOSC_STATUS  (XOSC_BASE + 0x04)
#define XOSC_STARTUP (XOSC_BASE + 0x0C)
#define CLK_REF_CTRL  (CLOCKS_BASE + 0x30)
#define CLK_PERI_CTRL (CLOCKS_BASE + 0x48)

static void clocks_init_xosc_12mhz(void)
{
    REG(XOSC_STARTUP) = 47;                                  /* ~1ms @12MHz */
    REG(XOSC_CTRL)    = (0xfabu << 12) | 0xaa0u;             /* ENABLE, 1-15MHz */
    while (!(REG(XOSC_STATUS) & (1u << 31))) { }             /* STABLE */
    REG(CLK_REF_CTRL)  = 2u;                                 /* clk_ref ← XOSC */
    REG(CLK_PERI_CTRL) = (1u << 11);                         /* clk_peri ← clk_sys */
}

static void uart_init_115200(void)
{
    REG(RESETS_CLR) = RESET_IO_BANK0 | RESET_PADS_BANK0 | RESET_UART0;
    while ((REG(RESETS_DONE) & (RESET_IO_BANK0 | RESET_PADS_BANK0 | RESET_UART0))
           != (RESET_IO_BANK0 | RESET_PADS_BANK0 | RESET_UART0)) { }
    REG(IO_BANK0_BASE + 0x004) = 2u;                         /* GP0 = UART0 TX */
    /* 12 MHz / (16 * 115200) = 6.5104; IBRD=6, FBRD=round(0.5104*64)=33 */
    REG(UART0_IBRD) = 6;
    REG(UART0_FBRD) = 33;
    REG(UART0_LCR_H) = UART_LCR_H_WLEN_8 | UART_LCR_H_FEN;
    REG(UART0_CR)    = UART_CR_UARTEN | UART_CR_TXE;
}

static void sh_puts(const char *s)
{
    while (*s) {
        while (REG(UART0_FR) & UART_FR_TXFF) { }
        REG(UART0_DR) = (uint32_t)(unsigned char)*s++;
    }
}

__attribute__((noreturn))
static void sh_exit(int code)
{
    (void)code;
    /* Drain the FIFO so the final marker actually leaves the chip. */
    while (REG(UART0_FR) & (1u << 3)) { }                    /* BUSY */
    while (1) { __asm__ volatile ("wfi"); }
}

/* ---- shared float/int formatters (copy of main_template.c) ---- */

static int fmt_int(char *buf, int v)
{
    char *p = buf;
    if (v < 0) { *p++ = '-'; v = -v; }
    char tmp[12];
    int n = 0;
    if (v == 0) { tmp[n++] = '0'; }
    else { while (v > 0) { tmp[n++] = (char)('0' + v % 10); v /= 10; } }
    while (n > 0) *p++ = tmp[--n];
    *p = 0;
    return (int)(p - buf);
}

static int fmt_float(char *buf, float v, int decimals)
{
    char *p = buf;
    if (v < 0.0f) { *p++ = '-'; v = -v; }
    int whole = (int)v;
    float frac = v - (float)whole;
    char tmp[12];
    int n = 0;
    if (whole == 0) { tmp[n++] = '0'; }
    else { while (whole > 0) { tmp[n++] = (char)('0' + whole % 10); whole /= 10; } }
    while (n > 0) *p++ = tmp[--n];
    if (decimals > 0) {
        *p++ = '.';
        for (int i = 0; i < decimals; i++) {
            frac *= 10.0f;
            int d = (int)frac;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            *p++ = (char)('0' + d);
            frac -= (float)d;
        }
    }
    *p = 0;
    return (int)(p - buf);
}

int main(void)
{
    clocks_init_xosc_12mhz();
    uart_init_115200();

    float X[T_LEN];
    float Y[T_LEN] = {0};
    char buf[32];

    sh_puts("=========================================\n");
    sh_puts("rc_predict on Pi Pico (Cortex-M0+, Wokwi)\n");
    sh_puts("=========================================\n");

    for (int i = 0; i < T_LEN; i++) X[i] = X_in[i];
    rc_predict((int64_t)T_LEN, X, Y);

    float sse = 0.0f, max_abs_diff = 0.0f;
    for (int i = 0; i < T_LEN; i++) {
        float d = Y[i] - Y_reference[i];
        sse += d * d;
        float ad = d < 0.0f ? -d : d;
        if (ad > max_abs_diff) max_abs_diff = ad;

        sh_puts("Step ");          fmt_int(buf, i);          sh_puts(buf);
        sh_puts(": Input=");       fmt_float(buf, X[i], 4);  sh_puts(buf);
        sh_puts(", Ref=");         fmt_float(buf, Y_reference[i], 4); sh_puts(buf);
        sh_puts(", Pred=");        fmt_float(buf, Y[i], 4);  sh_puts(buf);
        sh_puts("\n");
    }
    float mse = sse / (float)T_LEN;

    sh_puts("-----------------------------------------\n");
    sh_puts("MSE          : ");      fmt_float(buf, mse, 6);          sh_puts(buf);
    sh_puts("\nMax |diff|   : ");    fmt_float(buf, max_abs_diff, 6); sh_puts(buf);
    sh_puts("\n-----------------------------------------\n");
    sh_puts("EMULATOR_EXIT\n");
    sh_exit(0);
}
