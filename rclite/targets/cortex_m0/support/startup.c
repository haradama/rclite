/* Minimal Cortex-M0 startup for nRF51 / micro:bit under QEMU.
 *
 *   - Defines the vector table at the start of flash (sp + reset handler
 *     + system handlers; external IRQs left as 0 since we don't use them).
 *   - Reset_Handler copies .data from flash to RAM, zeroes .bss, calls main.
 */
#include <stdint.h>

extern uint32_t _sdata, _edata, _sidata, _sbss, _ebss, _stack_top;
extern int main(void);

__attribute__((noreturn)) static void Default_Handler(void)
{
    while (1) { /* spin */ }
}

__attribute__((noreturn)) void Reset_Handler(void)
{
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }
    (void)main();
    while (1) { /* spin if main returns */ }
}

__attribute__((section(".vector_table"), used))
const void *const vector_table[48] = {
    [0]  = (const void *)&_stack_top,
    [1]  = (const void *)Reset_Handler,
    [2]  = (const void *)Default_Handler,   /* NMI       */
    [3]  = (const void *)Default_Handler,   /* HardFault */
    [11] = (const void *)Default_Handler,   /* SVCall    */
    [14] = (const void *)Default_Handler,   /* PendSV    */
    [15] = (const void *)Default_Handler,   /* SysTick   */
};
