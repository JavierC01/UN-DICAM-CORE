import io
import math
from pathlib import Path

import pandas as pd

# Factores a newton y a segundos.
UNIDADES_CARGA = {"N": 1.0, "kN": 1000.0, "kgf": 9.80665, "tf": 9806.65, "lbf": 4.4482216152605}
UNIDADES_TIEMPO = {"s": 1.0, "ms": 0.001, "min": 60.0}
EXTENSIONES = (".csv", ".txt", ".xlsx")


class ImportacionCargaError(ValueError):
    pass


def leer_tabla(contenido: bytes, nombre_archivo: str) -> pd.DataFrame:
    extension = Path(nombre_archivo).suffix.lower()
    if extension not in EXTENSIONES:
        raise ImportacionCargaError(f"Formato no soportado ({extension}); usa {', '.join(EXTENSIONES)}")
    try:
        if extension == ".xlsx":
            tabla = pd.read_excel(io.BytesIO(contenido))
        else:
            # sep=None deja que pandas detecte ',', ';' o tabulador.
            tabla = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")
    except Exception as exc:
        raise ImportacionCargaError(f"No se pudo leer el archivo: {exc}") from exc
    if tabla.empty or len(tabla.columns) < 2:
        raise ImportacionCargaError("El archivo debe tener al menos dos columnas (tiempo y carga)")
    tabla.columns = [str(c).strip() for c in tabla.columns]
    return tabla


def _a_numero(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    # shortcut: la coma se toma como decimal (exportes en espanol); un separador de miles con coma se lee mal
    return pd.to_numeric(serie.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def extraer_serie(
    tabla: pd.DataFrame, columna_tiempo: str, columna_carga: str, unidad_tiempo: str, unidad_carga: str
) -> tuple[list[float], list[float], int]:
    """Devuelve (tiempos_s, cargas_n, filas_descartadas). Las filas no numericas se descartan y se cuentan."""
    for columna in (columna_tiempo, columna_carga):
        if columna not in tabla.columns:
            raise ImportacionCargaError(f"La columna '{columna}' no existe. Columnas: {', '.join(tabla.columns)}")
    if unidad_tiempo not in UNIDADES_TIEMPO:
        raise ImportacionCargaError(f"Unidad de tiempo no soportada: {unidad_tiempo}")
    if unidad_carga not in UNIDADES_CARGA:
        raise ImportacionCargaError(f"Unidad de carga no soportada: {unidad_carga}")

    tiempos = _a_numero(tabla[columna_tiempo]) * UNIDADES_TIEMPO[unidad_tiempo]
    cargas = _a_numero(tabla[columna_carga]) * UNIDADES_CARGA[unidad_carga]
    ok = tiempos.notna() & cargas.notna()
    tiempos, cargas = tiempos[ok].tolist(), cargas[ok].tolist()
    if len(tiempos) < 2:
        raise ImportacionCargaError("El archivo no tiene al menos dos filas numericas de tiempo y carga")
    if any(b <= a for a, b in zip(tiempos, tiempos[1:])):
        raise ImportacionCargaError("La columna de tiempo debe ser estrictamente creciente")
    if not all(math.isfinite(v) for v in tiempos + cargas):
        raise ImportacionCargaError("El archivo contiene valores infinitos")
    return tiempos, cargas, int((~ok).sum())
