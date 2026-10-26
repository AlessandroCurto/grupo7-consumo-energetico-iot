// Grupo 7 - Captura de datos de consumo energetico (ESP32 + INA219 + 3 LEDs)
// El ESP32 genera 6 clases de consumo con los LEDs y mide la corriente con el INA219.
// Envia por el puerto serie una linea por lectura:
//   ventana,clase,t_ms,voltaje_V,corriente_mA,potencia_mW
// Al terminar envia la palabra FIN.

#include <Wire.h>
#include <Adafruit_INA219.h>
#include "driver/gpio.h"

Adafruit_INA219 ina219;

// ---------- configuracion ----------
const int   N = 3;
const int   LEDS[N] = {27, 26, 25};          // D27 rojo, D26 naranja, D25 verde
const int   N_CLASES = 6;
const char* CLASES[N_CLASES] = {"apagado", "bajo", "medio", "alto", "parpadeo", "variable"};
const int   VENTANAS_POR_CLASE = 50;         // 50 x 6 = 300 ventanas (unos 12 minutos)
const int   DURACION_MS = 2000;              // cada ventana dura 2 segundos
const int   PERIODO_MS  = 20;                // una lectura cada 20 ms (50 por segundo)
const int   PAUSA_MS    = 300;               // descanso entre ventanas (no se graba)

// ---------- LEDs ----------
void ponerLeds(int mascara) {                // bit k = 1 -> LED k prendido
  for (int k = 0; k < N; k++) digitalWrite(LEDS[k], (mascara >> k) & 1 ? LOW : HIGH);
}
int mascaraAleatoria(int cuantos) {          // elige 'cuantos' LEDs al azar
  int m = 0;
  while (__builtin_popcount(m) < cuantos) m |= 1 << random(N);
  return m;
}

// ---------- una ventana de captura ----------
void capturarVentana(int ventana, int clase) {
  int fija = 0, periodo = 0, cambio = 0;
  if (clase == 1) fija = mascaraAleatoria(1);                 // bajo: 1 LED
  if (clase == 2) fija = mascaraAleatoria(2);                 // medio: 2 LEDs
  if (clase == 3) fija = 0b111;                               // alto: 3 LEDs
  if (clase == 4) { fija = mascaraAleatoria(1 + random(3));   // parpadeo: 1 a 3 LEDs
                    periodo = 200 + random(601); }            // periodo 200-800 ms
  if (clase == 5) cambio = 100 + random(201);                 // variable: cambia cada 100-300 ms

  int actual = 0, proximoCambio = 0;
  unsigned long t0 = millis();
  while (true) {
    unsigned long t = millis() - t0;
    if (t >= (unsigned long)DURACION_MS) break;

    int m = 0;
    if (clase >= 1 && clase <= 3) m = fija;
    if (clase == 4) m = ((t / (periodo / 2)) % 2 == 0) ? fija : 0;
    if (clase == 5) {
      if ((int)t >= proximoCambio) { actual = random(1 << N); proximoCambio = t + cambio; }
      m = actual;
    }
    ponerLeds(m);

    float v = ina219.getBusVoltage_V();
    float i = ina219.getCurrent_mA();
    float p = ina219.getPower_mW();
    Serial.printf("%d,%s,%lu,%.3f,%.2f,%.1f\n", ventana, CLASES[clase], t, v, i, p);

    unsigned long siguiente = (t / PERIODO_MS + 1) * PERIODO_MS;
    while (millis() - t0 < siguiente) delay(1);
  }
  ponerLeds(0);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(21, 22);
  for (int k = 0; k < N; k++) {
    digitalWrite(LEDS[k], HIGH);
    pinMode(LEDS[k], OUTPUT);
    digitalWrite(LEDS[k], HIGH);
    gpio_set_drive_capability((gpio_num_t)LEDS[k], GPIO_DRIVE_CAP_0);
  }
  if (!ina219.begin()) {
    Serial.println("ERROR: no se encontro el INA219");
    while (true) delay(100);
  }
  randomSeed(esp_random());

  Serial.println("# Grupo 7 - captura de consumo energetico");
  Serial.println("ventana,clase,t_ms,voltaje_V,corriente_mA,potencia_mW");

  int ventana = 0;
  for (int ronda = 0; ronda < VENTANAS_POR_CLASE; ronda++) {
    int orden[N_CLASES];
    for (int c = 0; c < N_CLASES; c++) orden[c] = c;
    for (int c = N_CLASES - 1; c > 0; c--) {                  // orden al azar en cada ronda
      int j = random(c + 1); int tmp = orden[c]; orden[c] = orden[j]; orden[j] = tmp;
    }
    for (int c = 0; c < N_CLASES; c++) {
      capturarVentana(ventana++, orden[c]);
      delay(PAUSA_MS);
    }
  }
  Serial.println("FIN");
}

void loop() {
  delay(1000);
}
