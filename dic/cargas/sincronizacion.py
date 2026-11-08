import math

import numpy as np

EVENTOS_INICIO = ("ninguno", "umbral_carga")


def instante_inicio(tiempos_s: list[float], cargas_n: list[float], evento_inicio: str, umbral_n: float | None) -> float:
    """Tiempo de la maquina que coincide con el frame de referencia (antes de sumar el offset)."""
    if evento_inicio == "ninguno":
        return tiempos_s[0]
    if evento_inicio == "umbral_carga":
        if umbral_n is None:
            raise ValueError("El evento 'umbral_carga' necesita umbral_n")
        for t, c in zip(tiempos_s, cargas_n):
            if abs(c) >= umbral_n:
                return t
        raise ValueError(f"La carga nunca alcanza el umbral de {umbral_n} N")
    raise ValueError(f"Evento de inicio desconocido: {evento_inicio}. Opciones: {', '.join(EVENTOS_INICIO)}")


def cargas_en_frames(
    tiempos_frame_s: list[float],
    tiempos_s: list[float],
    cargas_n: list[float],
    offset_s: float = 0.0,
    evento_inicio: str = "ninguno",
    umbral_n: float | None = None,
) -> tuple[list[float], int, int]:
    """Interpola la carga de la maquina al instante de cada frame (tiempo 0 = frame de referencia).

    Los frames que caen fuera del registro de la maquina quedan en NaN. Devuelve (cargas, interpolados, descartados).
    """
    t0 = instante_inicio(tiempos_s, cargas_n, evento_inicio, umbral_n) + offset_s
    t = np.asarray(tiempos_frame_s, dtype=np.float64) + t0
    dentro = (t >= tiempos_s[0]) & (t <= tiempos_s[-1])
    valores = np.interp(t, tiempos_s, cargas_n)
    cargas = [float(v) if d else math.nan for v, d in zip(valores, dentro)]
    return cargas, int(dentro.sum()), int((~dentro).sum())
