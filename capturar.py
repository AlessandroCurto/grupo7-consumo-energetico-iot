"""Grupo 7 - Guarda en un CSV los datos que envia el ESP32 por el puerto serie.

Uso:  python capturar.py COM4
(Cierra antes el Monitor Serie del Arduino IDE; dos programas no pueden usar el mismo puerto.)
"""
import sys
import time

import serial  # pip install pyserial

PUERTO = sys.argv[1] if len(sys.argv) > 1 else "COM4"
SALIDA = "datos_grupo7.csv"
CABECERA = "ventana,clase,t_ms,voltaje_V,corriente_mA,potencia_mW"


def es_dato(linea: str) -> bool:
    """True si la linea tiene la forma: entero,texto,entero,numero,numero,numero."""
    partes = linea.split(",")
    if len(partes) != 6:
        return False
    try:
        int(partes[0]); int(partes[2]); float(partes[3]); float(partes[4]); float(partes[5])
        return True
    except ValueError:
        return False


with serial.Serial(PUERTO, 115200, timeout=5) as esp, open(SALIDA, "w", encoding="utf-8") as f:
    time.sleep(0.5)
    esp.setDTR(False); esp.setRTS(True); time.sleep(0.2); esp.setRTS(False)   # reinicia el ESP32
    print(f"Leyendo {PUERTO}... (dura unos 12 minutos)")
    f.write(CABECERA + "\n")
    empezo, filas, ultima = False, 0, -1
    while True:
        linea = esp.readline().decode("utf-8", errors="ignore").strip()
        if not linea:
            continue
        if linea.startswith("ERROR"):
            print(linea); break
        if linea == "FIN":
            break
        if linea.startswith("#"):
            print(linea); continue
        if linea == CABECERA:            # desde aqui empiezan los datos
            empezo = True; continue
        if not empezo or not es_dato(linea):
            continue                     # ignora los mensajes de arranque del ESP32
        f.write(linea + "\n"); filas += 1
        ventana = int(linea.split(",")[0])
        if ventana != ultima and ventana % 10 == 0:
            print(f"  ventana {ventana} de 300")
        ultima = ventana

print(f"Listo: {filas} lecturas guardadas en {SALIDA}")
