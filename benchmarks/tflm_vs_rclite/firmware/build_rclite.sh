#!/usr/bin/env bash
# Build an rclite reservoir firmware for micro:bit (Cortex-M0) and report size.
# Uses the SAME toolchain as the TFLM firmware (TFLM's pinned arm-gcc) so the
# Flash/RAM comparison is apples-to-apples.  Arg 1: variant (i8 | i16).
set -euo pipefail
cd "$(dirname "$0")"
VARIANT=${1:-i8}
SRCDIR=rclite_$VARIANT

TFM=/tmp/tflite-micro
GCCBIN=$TFM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin
CC=$GCCBIN/arm-none-eabi-gcc
SIZE=$GCCBIN/arm-none-eabi-size
ARCH="-mcpu=cortex-m0 -mthumb"
CFLAGS="$ARCH -Os -std=c99 -ffunction-sections -fdata-sections -Wall"

mkdir -p build
$CC -c $CFLAGS -I. -I$SRCDIR $SRCDIR/rc_kernel.c -o build/rc_kernel_$VARIANT.o
$CC -c $CFLAGS -I. -I$SRCDIR main_rc.c           -o build/main_rc_$VARIANT.o
$CC -c $CFLAGS startup.c                          -o build/startup_rc.o

$CC $ARCH -T nrf51.ld -nostartfiles -Wl,--gc-sections \
    -Wl,-Map=build/rclite_$VARIANT.map --specs=nano.specs --specs=nosys.specs \
    build/startup_rc.o build/main_rc_$VARIANT.o build/rc_kernel_$VARIANT.o \
    -lc -lgcc -lnosys \
    -o build/rclite_$VARIANT.elf

$SIZE build/rclite_$VARIANT.elf
echo "ELF: $(pwd)/build/rclite_$VARIANT.elf"
