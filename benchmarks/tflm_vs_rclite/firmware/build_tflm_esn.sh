#!/usr/bin/env bash
# Build the TFLM ESN-cell firmware (same reservoir as rclite) for Cortex-M0.
set -euo pipefail
cd "$(dirname "$0")"

TFM=/tmp/tflite-micro
DL=$TFM/tensorflow/lite/micro/tools/make/downloads
GCCBIN=$DL/gcc_embedded/bin
CC=$GCCBIN/arm-none-eabi-gcc
CXX=$GCCBIN/arm-none-eabi-g++
SIZE=$GCCBIN/arm-none-eabi-size
LIB=$TFM/gen/cortex_m_generic_cortex-m0_release_gcc/lib
ARENA=${ARENA_SIZE:-8192}

INC="-I$TFM -I$DL -I$DL/gemmlowp -I$DL/flatbuffers/include -I$DL/kissfft -I$DL/ruy -I. -I../out"
DEF="-DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS -DARENA_SIZE=$ARENA"
ARCH="-mcpu=cortex-m0 -mthumb"
CXXFLAGS="$ARCH -Os -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-unwind-tables -ffunction-sections -fdata-sections"
CFLAGS="$ARCH -Os -ffunction-sections -fdata-sections"

mkdir -p build
$CXX -c $CXXFLAGS $DEF $INC main_tflm_esn.cc       -o build/main_tflm_esn.o
$CXX -c $CXXFLAGS $DEF $INC ../out/model_esn_data.cc -o build/model_esn_data.o
$CXX -c $CXXFLAGS $DEF $INC esn_test_data.cc       -o build/esn_test_data.o
$CC  -c $CFLAGS startup.c                           -o build/startup_esn.o

$CXX $ARCH -T nrf51.ld -nostartfiles -Wl,--gc-sections \
    -Wl,-Map=build/tflm_esn.map --specs=nano.specs --specs=nosys.specs \
    build/startup_esn.o build/main_tflm_esn.o build/model_esn_data.o build/esn_test_data.o \
    -L"$LIB" -ltensorflow-microlite \
    -lstdc++ -lsupc++ -lm -lc -lgcc -lnosys \
    -o build/tflm_esn.elf

$SIZE build/tflm_esn.elf
echo "ELF: $(pwd)/build/tflm_esn.elf"
