/* mGBA debug-logging interface for the GBA target's on-device test drivers.
 *
 * The GBA has no semihosting, so we use mGBA's MMIO log registers instead:
 *   write 0xC0DE to REG_DEBUG_ENABLE to turn logging on, fill REG_DEBUG_STRING
 *   with a NUL-terminated line (<=255 bytes), then write 0x100|level to
 *   REG_DEBUG_FLAGS to flush it. Level 3 = INFO, captured by `mgba -l 15`.
 */
#ifndef MGBA_LOG_H
#define MGBA_LOG_H
#include <stdint.h>

#define REG_DEBUG_ENABLE ((volatile uint16_t *)0x4FFF780)
#define REG_DEBUG_FLAGS  ((volatile uint16_t *)0x4FFF700)
#define REG_DEBUG_STRING ((volatile char *)0x4FFF600)
#define MGBA_LOG_INFO    3

static void mgba_open(void) { *REG_DEBUG_ENABLE = 0xC0DE; }

/* Emit one log line (truncated to 255 chars) at INFO level. */
static void mgba_log(const char *s)
{
    volatile char *d = REG_DEBUG_STRING;
    int i = 0;
    for (; i < 255 && s[i]; i++) d[i] = s[i];
    d[i] = 0;
    *REG_DEBUG_FLAGS = 0x100 | MGBA_LOG_INFO;
}

/* Line builder: assemble a line piecewise with ln_puts/ln_int, then ln_flush. */
static char _mgba_line[256];
static int  _mgba_n;

static void ln_reset(void) { _mgba_n = 0; _mgba_line[0] = 0; }

static void ln_puts(const char *s)
{
    for (int i = 0; s[i] && _mgba_n < 255; i++) _mgba_line[_mgba_n++] = s[i];
    _mgba_line[_mgba_n] = 0;
}

static void ln_int(int32_t v)
{
    char tmp[12]; int n = 0;
    char b[16];   int k = 0;
    if (v < 0) { b[k++] = '-'; v = -v; }
    if (v == 0) tmp[n++] = '0';
    else while (v > 0) { tmp[n++] = (char)('0' + v % 10); v /= 10; }
    while (n > 0) b[k++] = tmp[--n];
    b[k] = 0;
    ln_puts(b);
}

/* Append a Q-format fixed-point value with `decimals` fractional digits. */
__attribute__((unused))
static void ln_fixed(int32_t v, int frac_bits, int decimals)
{
    char b[24]; int k = 0;
    if (v < 0) { b[k++] = '-'; v = -v; }
    int32_t scale = (int32_t)1 << frac_bits;
    int32_t whole = v >> frac_bits;
    int32_t frac  = v & (scale - 1);
    char tmp[12]; int n = 0;
    if (whole == 0) tmp[n++] = '0';
    else while (whole > 0) { tmp[n++] = (char)('0' + whole % 10); whole /= 10; }
    while (n > 0) b[k++] = tmp[--n];
    if (decimals > 0) {
        b[k++] = '.';
        for (int i = 0; i < decimals; i++) {
            frac *= 10;
            int d = frac >> frac_bits;
            if (d < 0) d = 0; else if (d > 9) d = 9;
            b[k++] = (char)('0' + d);
            frac -= ((int32_t)d) << frac_bits;
        }
    }
    b[k] = 0;
    ln_puts(b);
}

/* Flush the accumulated line as one mGBA log entry, then reset the buffer. */
static void ln_flush(void) { mgba_log(_mgba_line); ln_reset(); }

#endif /* MGBA_LOG_H */
