/* Auto-generated rclite demo sketch for Arduino Uno (ATmega328P).
 *
 * Runs the quantized reservoir kernel on an embedded test sequence and
 * reports, over Serial @ 9600, the max |Y - Y_ref| in storage units
 * (0 == bit-exact with the host reference). The kernel itself is in
 * rc_kernel.c (pure integer, weights in Flash via PROGMEM).
 *
 * Placeholders filled in by ArduinoUnoTarget.compile_affine_quantized:
 *   @@T@@           number of inference steps
 *   @@STORAGE_T@@   int8_t / int16_t
 *   @@X_VALUES@@    embedded quantized inputs
 *   @@Y_VALUES@@    embedded quantized reference outputs
 */
#include <stdint.h>

/* rc_kernel.c is compiled as C; declare with C linkage from this C++ sketch. */
extern "C" void rc_predict(int32_t T, const @@STORAGE_T@@ *X, @@STORAGE_T@@ *Y);

#define RC_T @@T@@

static const @@STORAGE_T@@ X_q[RC_T] = { @@X_VALUES@@ };
static const @@STORAGE_T@@ Y_ref[RC_T] = { @@Y_VALUES@@ };
static @@STORAGE_T@@ Y_out[RC_T];

void setup() {
  Serial.begin(9600);
  while (!Serial) { /* wait for USB serial on some boards */ }

  @@STORAGE_T@@ X[RC_T];
  for (int i = 0; i < RC_T; i++) X[i] = X_q[i];

  unsigned long t0 = micros();
  rc_predict((int32_t)RC_T, X, Y_out);
  unsigned long dt = micros() - t0;

  int32_t max_abs_diff = 0;
  for (int t = 0; t < RC_T; t++) {
    int32_t d = (int32_t)Y_out[t] - (int32_t)Y_ref[t];
    if (d < 0) d = -d;
    if (d > max_abs_diff) max_abs_diff = d;
  }

  Serial.println(F("rclite affine kernel on Arduino Uno"));
  Serial.print(F("steps="));        Serial.println(RC_T);
  Serial.print(F("elapsed_us="));   Serial.println(dt);
  Serial.print(F("us_per_step="));  Serial.println(dt / (unsigned long)RC_T);
  Serial.print(F("max_abs_diff=")); Serial.println(max_abs_diff);
  if (max_abs_diff == 0) Serial.println(F("PARITY_OK"));
  else                   Serial.println(F("PARITY_FAIL"));
}

void loop() { }
