# Grupo 7 · Clasificación de consumo energético con sensores IoT

**Curso:** Machine Learning (UNI-FIIS) · **Profesor:** Dr. Ing. Hilario Aradiel Castañeda · **Ciclo:** 2026-II
**Modelo asignado:** Naive Bayes, comparado con otros 12 modelos.

| Integrante | Código |
|---|---|
| Crisostomo Cahuas, Gabriel Andre | 20240090I |
| Curto Meneses, Alessandro del Piero | 20242084F |
| Rubio Zeta, Luis Miguel | 202210132K |

## Descripción del proyecto

Simulamos a escala un medidor inteligente. Un **ESP32** prende y apaga 3 LEDs (rojo, naranja y verde) con distintos patrones de uso, y un sensor **INA219** mide el voltaje, la corriente y la potencia que consumen. Con esas lecturas entrenamos modelos que reconocen el **tipo de consumo** solo a partir de la señal de corriente.

**Clases de consumo (6):**

| Clase | Qué hacen los LEDs |
|---|---|
| `apagado` | Todos apagados |
| `bajo` | 1 LED prendido, elegido al azar |
| `medio` | 2 LEDs prendidos |
| `alto` | Los 3 LEDs prendidos |
| `parpadeo` | 1 a 3 LEDs que parpadean con un periodo de 200 a 800 ms |
| `variable` | Combinaciones al azar que cambian cada 100 a 300 ms |

## Archivos

| Archivo | Qué es |
|---|---|
| `captura_grupo7.ino` | Programa del ESP32: genera las clases y mide con el INA219 (50 lecturas por segundo) |
| `capturar.py` | Lee el puerto serie y guarda las lecturas en `datos_grupo7.csv` |
| `datos_grupo7.csv` | Datos captados: 300 capturas de 2 s (50 por clase), 30 000 lecturas |
| `main.py` | Preparación, modelado y evaluación |
| `main.ipynb` | El mismo código que `main.py`, por pasos, con las tablas y figuras dentro del cuaderno (Jupyter) |
| `requirements.txt` | Librerías necesarias |
| `resultados/` | Tablas (`tablas/`) y figuras (`figuras/`) que genera `main.py` |

## Hardware y conexión

- ESP32 DEVKIT V1, sensor INA219, protoboard MB-102, 3 LEDs y 3 resistencias de 22 Ω.
- INA219 → ESP32: Vcc → 3V3, Gnd → GND, Sda → D21, Scl → D22, Vin+ → VIN.
- Vin− del INA219 → riel + de la protoboard; ahí van las patas largas de los LEDs.
- Cada LED: pata corta → resistencia de 22 Ω → pin del ESP32 (D27 rojo, D26 naranja, D25 verde). El LED se prende cuando el pin pasa a nivel bajo. La corriente queda en unos 11 mA por LED, porque el pin trabaja con su fuerza de salida mínima.

## Fases trabajadas (CRISP-ML(Q))

1. **Comprensión de los datos:** carga del CSV y control de calidad (sin vacíos, 100 lecturas por captura, 50 capturas por clase).
2. **Preparación:** cada captura de 2 s se corta en **ventanas de 0,5 s** (25 lecturas). Cada ventana se resume en 17 variables: media, desviación, mínimo, máximo, percentiles, asimetría y curtosis de la corriente, fracción del tiempo con consumo, número de cambios bruscos, voltaje y potencia. Resultado: 1 200 ventanas.
3. **Descripción:** estadísticas por clase y **ranking de variables por ganancia de información** (información mutua).
4. **Modelado:** Naive Bayes y 12 modelos más:
   - regresión logística, k-NN, árbol de decisión;
   - bosque aleatorio, Extra Trees, Gradient Boosting, Hist. Gradient Boosting, AdaBoost;
   - SVM RBF, SVM lineal, LDA y red neuronal MLP.
5. **Optimización de parámetros:** búsqueda en rejilla con validación cruzada de 5 particiones, maximizando la exactitud balanceada.
6. **Evaluación:**
   - Se separan 70 % de las capturas para entrenar y 30 % para probar. La división se hace **por captura**, así que las ventanas de una misma captura nunca quedan a los dos lados y no hay fuga de información.
   - Métricas: exactitud, exactitud balanceada, valor predictivo positivo (VPP, precisión), F1 macro y pérdida logarítmica, más la matriz de confusión y la sensibilidad y el VPP de cada clase.
7. **Análisis adicional:** efecto del largo de la ventana (0,2 / 0,5 / 1 / 2 s).

## Resultados principales (ventanas de 0,5 s, prueba)

| Modelo | Exactitud balanceada | Pérdida logarítmica |
|---|---|---|
| Gradient Boosting (mejor) | 0,967 | 0,105 |
| Naive Bayes (asignado) | 0,936 | 1,439 |

Naive Bayes confunde sobre todo `variable` con `parpadeo`: las dos tienen la misma corriente promedio y solo se distinguen por la forma de la señal en el tiempo. Con ventanas de 2 s, Naive Bayes llega a 1,000.

## Cómo ejecutar, paso a paso

**A. Captura de datos** (solo si quieres volver a capturar; el CSV ya está incluido):

1. Arma el circuito y sube `captura_grupo7.ino` al ESP32 desde el Arduino IDE.
   - Placa: "ESP32 Dev Module".
   - Librería: "Adafruit INA219".
2. Cierra el Monitor Serie del Arduino IDE.
3. En una terminal, dentro de esta carpeta:
   ```
   pip install pyserial
   python capturar.py COM4
   ```
   Usa el COM que aparezca en tu laptop. Tarda unos 12 minutos y genera `datos_grupo7.csv`.

**B. Modelado y evaluación:**

1. Instala las librerías:
   ```
   pip install -r requirements.txt
   ```
2. Ejecuta:
   ```
   python main.py
   ```
   Tarda 1 a 2 minutos.
   O, si prefieres Jupyter, abre el cuaderno y ejecuta todas las celdas (Kernel → Restart & Run All):
   ```
   jupyter notebook main.ipynb
   ```
3. Revisa los resultados:
   - `resultados/tablas/comparacion_modelos.csv`: los 13 modelos con sus mejores parámetros y métricas.
   - `resultados/tablas/ranking_variables.csv`: la ganancia de información de cada variable.
   - `resultados/tablas/metricas_por_clase.csv`: sensibilidad (tasa de VP) y VPP de cada clase para Naive Bayes y el mejor modelo.
   - `resultados/figuras/fig01` a `fig07`: las señales, el ranking, la distribución de variables por clase, la correlación entre variables, la comparación de modelos, las matrices de confusión y el largo de ventana.

Con la semilla 42, dos corridas dan los mismos números.
