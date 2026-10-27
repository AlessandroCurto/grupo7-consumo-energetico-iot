// =====================================================================
// Grupo 7 - DEMO EN VIVO: el ESP32 genera los 6 tipos de consumo con los
// 3 LEDs, mide la corriente con el INA219 y Naive Bayes (entrenado en
// main.py) adivina la clase cada 1 segundo, dentro del mismo ESP32.
// Circuito: el mismo de la captura (D27 rojo, D26 naranja, D25 verde;
// INA219 por I2C en D21 = SDA y D22 = SCL).
// Uso: subir con Arduino IDE 2 (placa "ESP32 Dev Module") y abrir el
// Monitor Serie a 115200 baudios.
// =====================================================================
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <math.h>
#include "driver/gpio.h"

// true  -> Serial Plotter (grafica la corriente en vivo, sin texto)
// false -> Monitor Serie (texto: clase real y prediccion de Naive Bayes)
const bool MODO_GRAFICO = false;

Adafruit_INA219 ina219;
const int N = 3;
const int LEDS[N] = {27, 26, 25};              // D27 rojo, D26 naranja, D25 verde
const int ORDEN[6] = {1, 2, 3, 0, 4, 5};  // apagado, bajo, medio, alto, parpadeo, variable en CLASES_NB
const char* DESCRIPCION[6] = {"ningun LED", "1 LED fijo", "2 LEDs fijos", "3 LEDs fijos",
                              "LEDs que parpadean", "combinaciones que cambian al azar"};
const int VENTANAS_POR_CLASE = 4;              // 4 predicciones de 1 s por clase
const int LECTURAS = 50;                       // 1 s a 50 Hz
const int PERIODO_MS = 20;

// ---------- modelo Naive Bayes entrenado en Python (main.py, ventanas de 1 s) ----------
// Variables en el mismo orden que main.py: i_media, i_desv, i_min, i_max, i_mediana, i_rango, i_p10, i_p90, i_asimetria, i_curtosis, frac_encendido, n_cambios, dif_media, v_media, v_desv, p_media, p_desv
const int NV = 17;
const int NC = 6;
const char* CLASES_NB[NC] = {"alto", "apagado", "bajo", "medio", "parpadeo", "variable"};
const double MEDIA_ESC[NV] = {16.13649667, 3.291613432, 11.82933333, 20.0915, 17.05841667, 8.262166667, 12.30796667, 19.75885, -0.2279361372, 1.129380931, 0.7397333333, 1.341666667, 0.6486258503, 4.707278667, 0.005453993406, 75.762, 15.30527129};   // StandardScaler: media
const double ESCALA[NV]    = {10.69427219, 4.960864685, 12.22648789, 11.96952566, 11.53117441, 11.51892148, 12.36954787, 11.86624338, 1.319199629, 7.997961893, 0.3739713833, 2.142801256, 0.8848618816, 0.01108248821, 0.004364594971, 50.08796866, 23.40959254};   // StandardScaler: desviacion
const double THETA[NC][NV] = {
  {1.619826299, -0.6134535322, 1.683775983, 1.155392486, 1.422455575, -0.5866145264, 1.698383284, 1.169633014, -0.2334260457, 0.6060069941, 0.6959534292, -0.5701259803, -0.5423345343, -1.600964182, -0.3847857627, 1.625252574, -0.640336499},
  {-1.512901148, -0.63552344, -0.9961432456, -1.655830028, -1.483016045, -0.6632710087, -1.011845121, -1.654588515, 0.2059857708, -0.1527627269, -1.978047964, -0.6261274409, -0.5568877106, 1.549014356, -0.7057892037, -1.51157258, -0.6411042468},
  {-0.4466069858, -0.6311901005, -0.07641878366, -0.7023252418, -0.4937846279, -0.6486863099, -0.09039672899, -0.6927845435, -0.08260549934, 0.09567001633, 0.6959534292, -0.6261274409, -0.5464398677, 0.3876867047, -0.680417604, -0.4425014748, -0.6121741557},
  {0.6007761184, -0.6284753089, 0.8334909219, 0.2327159889, 0.4785794695, -0.6428697932, 0.8143331865, 0.2513221671, -0.1186255191, 0.06042355395, 0.6959534292, -0.6167938641, -0.5381600098, -0.5970019135, -0.5477769809, 0.5921981026, -0.6168196839},
  {-0.3856846539, 1.570953191, -0.9914812363, 0.2409034479, -0.02783035406, 1.302711661, -1.005822266, 0.2419257645, 0.07080780887, -0.3766010246, -0.5410396152, 1.506594848, 1.644355171, 0.4014742222, 1.560697712, -0.3868953068, 1.574200316},
  {0.1245903704, 0.9376891907, -0.4532236388, 0.7291433471, 0.1035959817, 1.238729977, -0.4046523542, 0.6844921129, 0.1578634846, -0.2327368128, 0.4312272913, 0.9325798778, 0.5394669512, -0.1402091875, 0.7580718391, 0.1235186845, 0.9362342699}};                                   // GaussianNB: media por clase
const double VARZ[NC][NV]  = {
  {0.01381807708, 0.004042503777, 0.0436020783, 0.001019470983, 0.01220428476, 0.04506952749, 0.01074002836, 0.01157717775, 3.418392741, 3.412811325, 1e-09, 0.0273543167, 0.004837598956, 0.01036534855, 0.1294465915, 0.01439824875, 0.001330881347},
  {3.50082074e-06, 8.486020952e-06, 3.679349547e-05, 2.803207957e-05, 1.51832262e-05, 6.719746463e-05, 1.036263914e-05, 1.328855158e-05, 0.06654026071, 0.01042368307, 1e-09, 1e-09, 0.0006491352288, 0.003366451409, 0.002898795897, 1.565027742e-06, 0.0001948141379},
  {0.001074169763, 0.0002081208243, 0.003388088117, 0.0008241468468, 0.0009582373123, 0.002973195852, 0.0008197086847, 0.0008530038034, 0.9720280479, 0.8846099072, 1e-09, 1e-09, 0.001209137714, 0.003037586632, 0.008639777079, 0.001161444978, 0.0007643378392},
  {0.0008962971367, 0.0006039020841, 0.006886218244, 0.0007298909457, 0.0007443230355, 0.00740134188, 0.0007367459663, 0.0007038813104, 1.003442481, 1.070727184, 1e-09, 0.0042686681, 0.001239563745, 0.003844621587, 0.0458811279, 0.001050611317, 0.0009775099036},
  {0.2236335412, 0.7747943617, 3.780361671e-05, 0.5648854086, 1.050930197, 0.610002935, 1.684376962e-05, 0.5605229837, 0.04907880533, 0.0002412969248, 0.03544567213, 0.7677284905, 1.470203571, 0.2057989431, 0.9924915576, 0.2232899507, 0.7674873645},
  {0.1232021413, 0.2995902496, 0.2296524428, 0.2187062105, 0.2280631049, 0.4848556469, 0.2326079225, 0.2504543482, 0.3427695057, 0.02180716899, 0.1201462138, 0.5715658142, 0.3344690844, 0.1235028357, 0.400965674, 0.1223495989, 0.2983346502}};                                     // GaussianNB: varianza por clase
const double LOG_PRIOR[NC] = {-1.791759469, -1.791759469, -1.791759469, -1.791759469, -1.791759469, -1.791759469};

// ---------- variables de una ventana (igual que caracteristicas_de_ventana en main.py) ----------
static void ordenar(double *a, int n) {
  for (int k = 1; k < n; k++) { double x = a[k]; int j = k - 1;
    while (j >= 0 && a[j] > x) { a[j + 1] = a[j]; j--; } a[j + 1] = x; }
}
static double percentil(const double *s, int n, double q) {      // como numpy.percentile (lineal)
  double idx = q / 100.0 * (n - 1); int lo = (int)idx; int hi = (lo + 1 < n) ? lo + 1 : lo;
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}
static double media(const double *a, int n) { double s = 0; for (int k = 0; k < n; k++) s += a[k]; return s / n; }
static double desv(const double *a, int n) {                       // desviacion poblacional (numpy std)
  double m = media(a, n), s = 0; for (int k = 0; k < n; k++) s += (a[k] - m) * (a[k] - m); return sqrt(s / n);
}

void caracteristicas(const double *i, const double *v, const double *p, int n, double *x) {
  static double s[64];
  for (int k = 0; k < n; k++) s[k] = i[k];
  ordenar(s, n);
  double mi = media(i, n), m2 = 0, m3 = 0, m4 = 0;
  for (int k = 0; k < n; k++) { double d = i[k] - mi; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
  m2 /= n; m3 /= n; m4 /= n;
  double sd = sqrt(m2);
  int encendido = 0, cambios = 0; double sumdif = 0;
  for (int k = 0; k < n; k++) if (i[k] > 5.0) encendido++;
  for (int k = 1; k < n; k++) { double d = fabs(i[k] - i[k - 1]); sumdif += d; if (d > 5.0) cambios++; }
  x[0]  = mi;                                   // i_media
  x[1]  = sd;                                   // i_desv
  x[2]  = s[0];                                 // i_min
  x[3]  = s[n - 1];                             // i_max
  x[4]  = percentil(s, n, 50);                  // i_mediana
  x[5]  = s[n - 1] - s[0];                      // i_rango
  x[6]  = percentil(s, n, 10);                  // i_p10
  x[7]  = percentil(s, n, 90);                  // i_p90
  x[8]  = sd > 1e-9 ? m3 / pow(m2, 1.5) : 0.0;  // i_asimetria
  x[9]  = sd > 1e-9 ? m4 / (m2 * m2) - 3.0 : 0.0; // i_curtosis
  x[10] = (double)encendido / n;                // frac_encendido
  x[11] = cambios;                              // n_cambios
  x[12] = sumdif / (n - 1);                     // dif_media
  x[13] = media(v, n);                          // v_media
  x[14] = desv(v, n);                           // v_desv
  x[15] = media(p, n);                          // p_media
  x[16] = desv(p, n);                           // p_desv
}

// ---------- Naive Bayes gaussiano: devuelve la clase con mayor probabilidad ----------
int predecir(const double *x, double *prob) {
  double ll[NC], mx = -1e300;
  for (int k = 0; k < NC; k++) {
    ll[k] = LOG_PRIOR[k];
    for (int j = 0; j < NV; j++) {
      double z = (x[j] - MEDIA_ESC[j]) / ESCALA[j];
      ll[k] += -0.5 * log(2.0 * M_PI * VARZ[k][j]) - (z - THETA[k][j]) * (z - THETA[k][j]) / (2.0 * VARZ[k][j]);
    }
    if (ll[k] > mx) mx = ll[k];
  }
  int mejor = 0; double suma = 0;
  for (int k = 0; k < NC; k++) suma += exp(ll[k] - mx);
  for (int k = 0; k < NC; k++) if (ll[k] > ll[mejor]) mejor = k;
  *prob = 1.0 / suma;                           // probabilidad de la clase elegida
  return mejor;
}

// ---------- LEDs (nivel bajo = LED prendido, como en la captura) ----------
void ponerLeds(int mascara) {
  for (int k = 0; k < N; k++) digitalWrite(LEDS[k], (mascara >> k) & 1 ? LOW : HIGH);
}
int mascaraAleatoria(int cuantos) {
  int m = 0;
  while (__builtin_popcount(m) < cuantos) m |= 1 << random(N);
  return m;
}

double bufI[LECTURAS], bufV[LECTURAS], bufP[LECTURAS];
int aciertos = 0, total = 0;

void demoClase(int c) {                        // c: 0 apagado ... 5 variable
  const char* real = CLASES_NB[ORDEN[c]];
  int fija = 0, periodo = 0, cambio = 0;
  if (c == 1) fija = mascaraAleatoria(1);
  if (c == 2) fija = mascaraAleatoria(2);
  if (c == 3) fija = 0b111;
  if (c == 4) { fija = mascaraAleatoria(1 + random(3)); periodo = 200 + random(601); }
  if (c == 5) cambio = 100 + random(201);
  if (!MODO_GRAFICO) Serial.printf("\n=== Clase real: %s (%s) ===\n", real, DESCRIPCION[c]);

  int actual = 0; unsigned long proximoCambio = 0;
  unsigned long t0 = millis();
  for (int w = 0; w < VENTANAS_POR_CLASE; w++) {
    for (int s = 0; s < LECTURAS; s++) {
      unsigned long t = millis() - t0;
      int m = 0;
      if (c >= 1 && c <= 3) m = fija;
      if (c == 4) m = ((t / (periodo / 2)) % 2 == 0) ? fija : 0;
      if (c == 5) { if (t >= proximoCambio) { actual = random(1 << N); proximoCambio = t + cambio; } m = actual; }
      ponerLeds(m);
      bufV[s] = ina219.getBusVoltage_V();
      bufI[s] = ina219.getCurrent_mA();
      bufP[s] = ina219.getPower_mW();
      if (MODO_GRAFICO) Serial.printf("corriente_mA:%.2f\n", bufI[s]);
      unsigned long siguiente = (unsigned long)(w * LECTURAS + s + 1) * PERIODO_MS;
      while (millis() - t0 < siguiente) delay(1);
    }
    double x[NV], prob;
    caracteristicas(bufI, bufV, bufP, LECTURAS, x);
    int k = predecir(x, &prob);
    bool ok = (k == ORDEN[c]);
    aciertos += ok; total++;
    if (!MODO_GRAFICO)
      Serial.printf("  seg %d | I media %5.1f mA | cambios %2d | Naive Bayes: %-9s (%3.0f %%)  %s\n",
                    w + 1, fabs(x[0]) < 0.05 ? 0.0 : x[0], (int)x[11], CLASES_NB[k], 100.0 * prob, ok ? "ACIERTO" : "ERROR");
  }
  ponerLeds(0);
  delay(800);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(21, 22);
  for (int k = 0; k < N; k++) {
    digitalWrite(LEDS[k], HIGH);
    pinMode(LEDS[k], OUTPUT);
    digitalWrite(LEDS[k], HIGH);
    gpio_set_drive_capability((gpio_num_t)LEDS[k], GPIO_DRIVE_CAP_0);   // limita la corriente (~11 mA por LED)
  }
  if (!ina219.begin()) {
    Serial.println("ERROR: no se encontro el INA219. Revisa Vcc, Gnd, Sda y Scl.");
    while (true) delay(100);
  }
  randomSeed(esp_random());
  if (!MODO_GRAFICO) {
    Serial.println("\nGrupo 7 - Demo: clasificacion de consumo energetico con Naive Bayes dentro del ESP32");
    Serial.println("Cada segundo el ESP32 mide 50 lecturas, calcula 17 variables y predice la clase.");
  }
}

void loop() {
  aciertos = 0; total = 0;
  for (int c = 0; c < 6; c++) demoClase(c);
  if (!MODO_GRAFICO)
    Serial.printf("\n>>> Resultado de la vuelta: %d de %d aciertos (%.0f %%)\n", aciertos, total, 100.0 * aciertos / total);
  delay(3000);
}
