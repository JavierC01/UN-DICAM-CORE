"""Calibracion px <-> mm: el usuario marca dos puntos cuya distancia real conoce y la ingresa en mm."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from dic.consola import pedir_float
from dic.interactivo import confirmar, marcar_puntos

DISTANCIA_MINIMA_PX = 20  # por debajo, un error de 1 px al marcar ya es >5 % de error de escala


@dataclass(frozen=True)
class Calibracion:
    puntos_px: list[dict[str, float]]
    distancia_px: float
    distancia_mm: float

    @property
    def px_por_mm(self) -> float:
        return self.distancia_px / self.distancia_mm

    @property
    def mm_por_px(self) -> float:
        return self.distancia_mm / self.distancia_px


def dibujar_calibracion(frame: np.ndarray, cal: Calibracion) -> np.ndarray:
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    (a, b) = [(int(p["x"]), int(p["y"])) for p in cal.puntos_px]
    grosor = max(1, int(max(img.shape[:2]) / 600))
    cv2.line(img, a, b, (0, 255, 255), grosor)
    for p in (a, b):
        cv2.circle(img, p, 3 * grosor, (0, 0, 255), -1)
    medio = ((a[0] + b[0]) // 2 + 8, (a[1] + b[1]) // 2 - 8)
    texto = f"{cal.distancia_mm:g} mm = {cal.distancia_px:.1f} px  ({cal.px_por_mm:.3f} px/mm)"
    cv2.putText(img, texto, medio, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * grosor, (0, 0, 0), 3 * grosor)
    cv2.putText(img, texto, medio, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * grosor, (0, 255, 255), grosor)
    return img


def calibrar(frame: np.ndarray) -> Calibracion:
    """Repite hasta que el usuario acepta la calibracion."""
    while True:
        print("\n   Marca dos puntos del video cuya distancia real conozcas (marcas de la probeta, regla, mordazas...).")
        a, b = marcar_puntos(frame, "Calibracion: marca los 2 puntos de distancia conocida", n_puntos=2)
        distancia_px = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        print(f"   Distancia marcada: {distancia_px:.1f} px")
        if distancia_px < DISTANCIA_MINIMA_PX:
            print(f"   Los puntos estan a menos de {DISTANCIA_MINIMA_PX} px: elige puntos mas separados.")
            continue
        distancia_mm = pedir_float("Distancia real entre esos dos puntos (mm)")
        cal = Calibracion(puntos_px=[a, b], distancia_px=round(distancia_px, 2), distancia_mm=distancia_mm)
        print(f"   Escala: {cal.px_por_mm:.4f} px/mm  |  {cal.mm_por_px:.5f} mm/px")
        if confirmar(dibujar_calibracion(frame, cal), "Calibracion"):
            return cal
