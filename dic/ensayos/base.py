"""Interfaz comun de los tipos de ensayo.

Cada ensayo define su extensometro virtual (medida, eje x), la respuesta a partir de la carga (eje y) y la
propiedad que resulta de la pendiente de esa curva. Un ensayo nuevo se agrega aqui sin tocar el pipeline DIC.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

Punto = tuple[float, float]


@dataclass(frozen=True)
class Eje:
    nombre: str
    unidad: str


class Ensayo(ABC):
    tipo: str
    nombre: str
    n_puntos: int
    puntos: str  # que marcar sobre el frame, en orden
    medida: Eje
    respuesta: Eje
    propiedad: Eje

    def validar_parametros(self, parametros: dict) -> None:
        """Lanza ValueError si faltan parametros propios del ensayo."""

    def validar_puntos(self, puntos: list[dict] | None) -> None:
        if not puntos or len(puntos) != self.n_puntos:
            raise ValueError(f"El ensayo de {self.nombre} necesita {self.n_puntos} puntos: {self.puntos}")

    @abstractmethod
    def medir(self, puntos_mm: list[Punto], desplazamientos_mm: list[Punto], parametros: dict) -> float:
        """Lectura del extensometro virtual; NaN si algun punto no tiene desplazamiento."""

    def calcular_respuesta(self, carga_n: float, area_mm2: float, parametros: dict) -> float:
        return carga_n

    def describir(self) -> dict:
        return {
            "tipo": self.tipo,
            "nombre": self.nombre,
            "n_puntos": self.n_puntos,
            "puntos": self.puntos,
            "medida": vars(self.medida),
            "respuesta": vars(self.respuesta),
            "propiedad": vars(self.propiedad),
        }


def _longitud(a: Punto, b: Punto) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def deformados(puntos_mm: list[Punto], desplazamientos_mm: list[Punto]) -> list[Punto]:
    return [(p[0] + d[0], p[1] + d[1]) for p, d in zip(puntos_mm, desplazamientos_mm)]


def ajuste_lineal(
    x: list[float | None],
    y: list[float | None],
    x_min: float | None = None,
    x_max: float | None = None,
    fraccion_min: float | None = None,
    fraccion_max: float | None = None,
) -> dict:
    """Recta por minimos cuadrados sobre el rango que eligio el usuario (en x, o como fraccion de la respuesta maxima,

    p. ej. 0.1-0.4 Fmax). Nunca elige el rango por su cuenta: sin rango lanza ValueError.
    """
    xa = np.array([math.nan if v is None else v for v in x], dtype=np.float64)
    ya = np.array([math.nan if v is None else v for v in y], dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    if x_min is not None and x_max is not None:
        if x_min >= x_max:
            raise ValueError("El limite inferior del rango debe ser menor que el superior")
        ok &= (xa >= x_min) & (xa <= x_max)
        rango = {"tipo": "x", "min": x_min, "max": x_max}
    elif fraccion_min is not None and fraccion_max is not None:
        if not 0 <= fraccion_min < fraccion_max <= 1:
            raise ValueError("Las fracciones deben cumplir 0 <= minima < maxima <= 1")
        y_max = np.nanmax(np.where(ok, ya, np.nan)) if ok.any() else math.nan
        if not np.isfinite(y_max) or y_max <= 0:
            raise ValueError("No hay respuesta positiva para calcular fracciones de su maximo")
        ok &= (ya >= fraccion_min * y_max) & (ya <= fraccion_max * y_max)
        rango = {"tipo": "fraccion_respuesta_max", "min": fraccion_min, "max": fraccion_max, "respuesta_max": float(y_max)}
    else:
        raise ValueError("Indica el rango del ajuste: x_min/x_max o fraccion_min/fraccion_max")

    if ok.sum() < 3:
        raise ValueError(f"El rango elegido contiene {int(ok.sum())} puntos validos; se necesitan al menos 3")
    xs, ys = xa[ok], ya[ok]
    if np.ptp(xs) == 0:
        raise ValueError("Todos los puntos del rango tienen la misma medida; no se puede ajustar una recta")
    pendiente, intercepto = np.polyfit(xs, ys, 1)
    residuo = ys - (pendiente * xs + intercepto)
    total = np.sum((ys - ys.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residuo**2) / total) if total > 0 else math.nan
    return {"pendiente": float(pendiente), "intercepto": float(intercepto), "r2": r2, "n_puntos": int(ok.sum()), "rango": rango}
