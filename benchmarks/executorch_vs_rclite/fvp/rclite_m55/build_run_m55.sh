#!/usr/bin/env bash
# Build the rclite ESN (affine i8) portable-C kernel as a bare-metal firmware
# for the Arm Corstone-300 (Cortex-M55) — the SAME FVP subsystem ExecuTorch's
# runner targets — and run it on FVP_Corstone_SSE-300_Ethos-U55 (CPU only; no
# NPU). Reports the firmware ELF size + a SYS_ELAPSED instruction estimate.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../../.. && pwd)
ETDIR=${ET_DIR:-/tmp/executorch}
GCCBIN=$ETDIR/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin
CC=$GCCBIN/arm-none-eabi-gcc
SIZE=$GCCBIN/arm-none-eabi-size
FW=$REPO/benchmarks/tflm_vs_rclite/firmware     # main_rc.c, syshelp.h, rclite_i8/
KDIR=$FW/rclite_i8                              # rc_kernel.c, rc_data.h (gen_rclite_fw.py)
LIBPY39_DIR=${LIBPY39_DIR:-$(dirname "$(find "$HOME/.local/share/uv/python" -name 'libpython3.9.so' 2>/dev/null | head -1)")}

ARCH="-mcpu=cortex-m55 -mthumb"
CFLAGS="$ARCH -Os -std=c99 -ffunction-sections -fdata-sections -Wall"
mkdir -p build
$CC -c $CFLAGS -I. -I"$FW" -I"$KDIR" "$KDIR/rc_kernel.c" -o build/rc_kernel.o
$CC -c $CFLAGS -I. -I"$FW" -I"$KDIR" main_rc_m55.c       -o build/main_rc.o
$CC -c $CFLAGS startup.c                                  -o build/startup.o
$CC $ARCH -T m55_sse300.ld -nostartfiles -Wl,--gc-sections \
    --specs=nano.specs --specs=nosys.specs \
    build/startup.o build/main_rc.o build/rc_kernel.o -lc -lgcc -lnosys \
    -o build/rclite_m55.elf

echo "=== rclite ESN i8 firmware on Cortex-M55 ==="
$SIZE build/rclite_m55.elf

FVP=$ETDIR/examples/arm/arm-scratch/FVP-corstone300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55
echo "=== run on Corstone-300 FVP (Cortex-M55, CPU) ==="
LD_LIBRARY_PATH="$LIBPY39_DIR:${LD_LIBRARY_PATH-}" timeout 120 "$FVP" \
    -a build/rclite_m55.elf \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C cpu0.semihosting-enable=1 \
    --timelimit 60 2>&1 | sed 's/\r$//' | grep -vE "^$" | tail -20
