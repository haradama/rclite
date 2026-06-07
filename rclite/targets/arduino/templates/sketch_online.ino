/* Auto-generated rclite ONLINE-learning demo sketch for Arduino Uno (ATmega328P).
 *
 * On-device integer LMS / NLMS readout adaptation. The reservoir + the INITIAL
 * readout are baked into rc_kernel.c (const tables in Flash via PROGMEM, the
 * mutable rc_W_out in SRAM). This harness streams an embedded (input, target)
 * sequence through rc_train_step / rc_infer_step exactly as the host reference
 * `IntegerLMSLearner` does, then checks — over Serial @ 9600 — that:
 *
 *   * every per-step prediction matches the embedded reference  (max_pred_diff)
 *   * the final learned rc_W_out matches the reference checkpoint (max_w_diff)
 *
 * 0 / 0 == bit-exact with the host. Also reports us/step (note: AVR emulates
 * the kernel's int64 math in software, so NLMS's per-step divide is the costly
 * part). Embedded reference data lives in Flash (PROGMEM) to spare the 2 KB SRAM.
 *
 * Placeholders filled by ArduinoUnoTarget.compile_symmetric_online:
 *   @@T@@ @@RC_K@@ @@RC_M@@ @@RC_F@@ @@STORAGE_T@@ @@PGM_RD@@
 *   @@U_VALUES@@   pre-quantized inputs, row-major (T, K), input scale
 *   @@YT_VALUES@@  pre-quantized targets, row-major (T, M), state scale (int32)
 *   @@WARM_VALUES@@  per-step flag (1 = inference-only / warmup, 0 = train)
 *   @@YP_REF@@     reference per-step predictions, row-major (T, M) (int32)
 *   @@WOUT_REF@@   reference final readout, row-major (M, F) (int32)
 */
#include <stdint.h>
#include <avr/pgmspace.h>

extern "C" {
  void rc_train_reset(void);
  void rc_infer_step(const @@STORAGE_T@@ *u_q, int32_t *y_pred_q);
  void rc_train_step(const @@STORAGE_T@@ *u_q, const int32_t *y_target_q,
                     int32_t *y_pred_q);
  void rc_export_W_out(int32_t *dst);
}

#define RC_T @@T@@
#define RC_K @@RC_K@@
#define RC_M @@RC_M@@
#define RC_F @@RC_F@@

static const @@STORAGE_T@@ U_q[RC_T * RC_K]  PROGMEM = { @@U_VALUES@@ };
static const int32_t       YT_q[RC_T * RC_M] PROGMEM = { @@YT_VALUES@@ };
static const uint8_t       WARM[RC_T]        PROGMEM = { @@WARM_VALUES@@ };
static const int32_t       YP_ref[RC_T * RC_M] PROGMEM = { @@YP_REF@@ };
static const int32_t       W_ref[RC_M * RC_F]  PROGMEM = { @@WOUT_REF@@ };

static @@STORAGE_T@@ u_buf[RC_K];
static int32_t yt_buf[RC_M];
static int32_t yp[RC_M];
static int32_t W_out_buf[RC_M * RC_F];

void setup() {
  Serial.begin(9600);
  while (!Serial) { /* wait for USB serial on some boards */ }

  int32_t max_pred_diff = 0;
  int32_t max_w_diff = 0;
  int t, k, m, i;

  rc_train_reset();
  unsigned long t0 = micros();
  for (t = 0; t < RC_T; t++) {
    for (k = 0; k < RC_K; k++)
      u_buf[k] = (@@STORAGE_T@@)pgm_read_@@PGM_RD@@(&U_q[t * RC_K + k]);
    if (pgm_read_byte(&WARM[t])) {
      rc_infer_step(u_buf, yp);
    } else {
      for (m = 0; m < RC_M; m++)
        yt_buf[m] = (int32_t)pgm_read_dword(&YT_q[t * RC_M + m]);
      rc_train_step(u_buf, yt_buf, yp);
    }
    for (m = 0; m < RC_M; m++) {
      int32_t r = (int32_t)pgm_read_dword(&YP_ref[t * RC_M + m]);
      int32_t d = yp[m] - r; if (d < 0) d = -d;
      if (d > max_pred_diff) max_pred_diff = d;
    }
  }
  unsigned long dt = micros() - t0;

  rc_export_W_out(W_out_buf);
  for (i = 0; i < RC_M * RC_F; i++) {
    int32_t r = (int32_t)pgm_read_dword(&W_ref[i]);
    int32_t d = W_out_buf[i] - r; if (d < 0) d = -d;
    if (d > max_w_diff) max_w_diff = d;
  }

  Serial.println(F("rclite online (integer LMS/NLMS) kernel on Arduino Uno"));
  Serial.print(F("steps="));          Serial.println(RC_T);
  Serial.print(F("elapsed_us="));     Serial.println(dt);
  Serial.print(F("us_per_step="));    Serial.println(dt / (unsigned long)RC_T);
  Serial.print(F("max_pred_diff="));  Serial.println(max_pred_diff);
  Serial.print(F("max_w_diff="));     Serial.println(max_w_diff);
  if (max_pred_diff == 0 && max_w_diff == 0) Serial.println(F("PARITY_OK"));
  else                                       Serial.println(F("PARITY_FAIL"));
}

void loop() { }
