# UN-DICAM CORE

Herramienta DIC 2D (correlación digital de imágenes) para ensayos mecánicos. Todo se ejecuta desde `main.py`.

## Instalación (una sola vez)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Solo para carga por OCR: instalar [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) y tenerlo en el PATH.

## Uso

```powershell
.venv\Scripts\python main.py
```

El asistente pregunta paso a paso (Enter acepta el valor entre `[corchetes]`):

1. **Identificación de la probeta** (ej. `P-01`) y descripción opcional. Da nombre a la carpeta de resultados y al reporte.
2. **Fuente de imágenes y datos por segundo**: la herramienta detecta si el equipo tiene cámara. Si no hay ninguna,
   la opción "Cámara en vivo" aparece tachada y no se puede elegir.
   - *Archivo de video*: número de la lista de `videos/`, nombre del archivo (con o sin extensión) o ruta completa;
     inicio y fin del análisis en segundos, y **cuántos datos por segundo** se necesitan. El video se descompone
     tomando 1 de cada N frames (p. ej. video a 30 fps y 10 datos/s → 1 de cada 3 frames). Si el cociente no es
     exacto se usa el valor más cercano y se informa.
   - *Cámara en vivo*: se elige la cámara (si hay varias), los datos por segundo y la duración. La captura arranca al
     final del asistente (Enter), se guarda en memoria con el tiempo real de cada imagen y después se analiza.
3. **Tipo de ensayo**: tracción, compresión, flexión, cortante o pórtico.
4. **Medidas de la probeta**: ancho, espesor y longitud de calibre en mm.
5. **Calibración**: marca con el mouse dos puntos cuya distancia real conozcas y escribe esa distancia en mm.
   La herramienta calcula px/mm y muestra la línea para confirmar.
6. **Identificación de la probeta en el video**: contornea la zona a analizar, elige filas × columnas de nodos y
   revisa la vista previa de la malla (nodos verdes = válidos, rojos = fuera de la zona).
7. **Extensómetro virtual**: marca los puntos que pide el ensayo, dentro de la malla.
8. **Carga**: ninguna, archivo de la máquina (`cargas/`, CSV/TXT/XLSX) u OCR de la pantalla.
9. **Parámetros**: rango de búsqueda y rango del ajuste lineal para la propiedad (p. ej. módulo de elasticidad).

En las ventanas: clic izquierdo agrega un punto, clic derecho o Backspace deshace, Enter acepta, R repite la vista previa
y Esc cancela.

Resultados en `resultados/<id>_<fecha>/`: `resultados.csv`, `resultados.json`, `reporte.pdf`, `resumen.jpg`,
mapas `mapa_xx/yy/xy.jpg` y `config_usada.json`. Para repetir un análisis sin volver a marcar nada:

```powershell
.venv\Scripts\python main.py --config resultados\P-01_20261108_120000\config_usada.json
```

## Estructura

| Archivo | Contenido |
|---|---|
| `main.py` | Asistente, ejecución y exportación |
| `dic/calibracion.py` | Calibración px ↔ mm con dos puntos de distancia conocida |
| `dic/mallado.py` | Identificación de la probeta y generación de la malla con vista previa |
| `dic/analisis.py` | Análisis completo del video |
| `dic/fuentes.py` | Detección de cámaras, descomposición del video a N datos/s y captura en vivo |
| `dic/correlation.py`, `displacement.py`, `pipeline.py` | Motor DIC: ZNCC (numba), búsqueda y refinamiento subpíxel |
| `dic/rastreo.py`, `strain.py`, `probeta.py` | Malla, triangulación, deformaciones CST, escala y esfuerzo |
| `dic/ensayos/` | Tipos de ensayo y ajuste lineal |
| `dic/cargas/` | Importación y sincronización de la carga de la máquina |
| `dic/ocr.py` | Lectura de carga por OCR |
| `dic/mapas.py`, `export.py` | Mapas de deformación y exportación CSV/PDF/JPG |
| `dic/interactivo.py`, `consola.py` | Ventanas de marcación y preguntas por consola |
| `videos/`, `cargas/` | Carpetas sugeridas para los videos y los archivos de la máquina |
