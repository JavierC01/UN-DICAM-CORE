"""Geometria de la malla de rastreo: mascara del poligono, nodos validos y triangulacion.

Es la unica implementacion de la distribucion de nodos; services/tracking.py la usa.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib.tri import Triangulation


@dataclass
class MallaRastreo:
    nodos: list[dict[str, float]]
    validos: list[bool]
    poligono: list[dict[str, float]]
    area_px: float
    mascara: np.ndarray


def generar_mascara(ancho: int, alto: int, puntos: list[dict[str, float]]) -> np.ndarray:
    poligono = np.array([[(int(round(p["x"])), int(round(p["y"]))) for p in puntos]], dtype=np.int32)
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.fillPoly(mascara, poligono, 255)
    return mascara


def erosionar_por_subset(mascara: np.ndarray, subset_size: int) -> np.ndarray:
    """Pixeles donde un subset de subset_size x subset_size centrado (como extraer_subset) cae entero en la mascara."""
    half = subset_size // 2
    kernel = np.ones((subset_size, subset_size), dtype=np.uint8)
    return cv2.erode(mascara, kernel, anchor=(half, half), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def nodos_validos(mascara: np.ndarray, nodos: list[dict[str, float]], subset_size: int) -> list[bool]:
    erosionada = erosionar_por_subset(mascara, subset_size)
    alto, ancho = erosionada.shape
    validos = []
    for nodo in nodos:
        x, y = int(round(nodo["x"])), int(round(nodo["y"]))
        validos.append(bool(0 <= x < ancho and 0 <= y < alto and erosionada[y, x] > 0))
    return validos


def construir_malla(
    ancho: int, alto: int, puntos: list[dict[str, float]], filas: int = 3, columnas: int = 3, subset_size: int = 32
) -> MallaRastreo:
    """Grilla filas x columnas (ordenada por filas) sobre el rectangulo envolvente de la zona valida.

    La zona valida es la mascara erosionada por el subset; los nodos que caen fuera de ella (huecos de una
    forma en L, entallas) quedan marcados como invalidos en lugar de forzarse al borde.
    """
    mascara = generar_mascara(ancho, alto, puntos)
    erosionada = erosionar_por_subset(mascara, subset_size)
    ys, xs = np.where(erosionada > 0)
    if len(xs) == 0:
        raise ValueError("La region marcada es demasiado pequena para el tamano de subset elegido")

    nodos = [
        {"x": float(round(x)), "y": float(round(y))}
        for y in np.linspace(ys.min(), ys.max(), filas)
        for x in np.linspace(xs.min(), xs.max(), columnas)
    ]
    validos = nodos_validos(mascara, nodos, subset_size)
    if sum(validos) < 3:
        raise ValueError("La malla necesita al menos 3 nodos validos dentro de la region; aumenta filas/columnas")
    return MallaRastreo(nodos=nodos, validos=validos, poligono=puntos, area_px=area_poligono(puntos), mascara=mascara)


def area_poligono(puntos: list[dict[str, float]]) -> float:
    x = np.array([p["x"] for p in puntos])
    y = np.array([p["y"] for p in puntos])
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def centroide_poligono(puntos: list[dict[str, float]]) -> tuple[float, float]:
    x = np.array([p["x"] for p in puntos], dtype=np.float64)
    y = np.array([p["y"] for p in puntos], dtype=np.float64)
    cruz = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cruz.sum() / 2
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + np.roll(x, -1)) * cruz).sum() / (6 * area)), float(((y + np.roll(y, -1)) * cruz).sum() / (6 * area))


def indice_nodo_central(nodos: list[dict[str, float]], validos: list[bool], poligono: list[dict[str, float]]) -> int:
    """Nodo valido mas cercano al centroide del poligono."""
    cx, cy = centroide_poligono(poligono)
    candidatos = [i for i, v in enumerate(validos) if v]
    if not candidatos:
        raise ValueError("La malla no tiene nodos validos")
    return min(candidatos, key=lambda i: (nodos[i]["x"] - cx) ** 2 + (nodos[i]["y"] - cy) ** 2)


def triangular(nodos: list[dict[str, float]], validos: list[bool], mascara: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulacion de Delaunay de los nodos validos, sin triangulos que crucen fuera de la forma.

    Devuelve triangulos con indices de la lista original de nodos.
    """
    indices = [i for i, v in enumerate(validos) if v]
    if len(indices) < 3:
        return []
    xs = np.array([nodos[i]["x"] for i in indices])
    ys = np.array([nodos[i]["y"] for i in indices])
    try:
        tri = Triangulation(xs, ys)
    except (ValueError, RuntimeError):
        return []  # nodos colineales

    alto, ancho = mascara.shape
    dentro = lambda x, y: 0 <= int(round(x)) < ancho and 0 <= int(round(y)) < alto and mascara[int(round(y)), int(round(x))] > 0
    triangulos = []
    for a, b, c in tri.triangles:
        pa, pb, pc = (xs[a], ys[a]), (xs[b], ys[b]), (xs[c], ys[c])
        area = 0.5 * abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]))
        if area < 1e-6:
            continue
        # shortcut: se revisan centroide y puntos medios de aristas; una arista que roce un hueco muy angosto podria pasar
        muestras = [((pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3)]
        muestras += [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in ((pa, pb), (pb, pc), (pc, pa))]
        if all(dentro(x, y) for x, y in muestras):
            triangulos.append((indices[a], indices[b], indices[c]))
    return triangulos


@dataclass(frozen=True)
class GeometriaMalla:
    """Lo que el pipeline necesita de la malla, calculado una vez por analisis."""

    nodos: list[dict[str, float]]
    validos: list[bool]
    triangulos: list[tuple[int, int, int]]
    indice_centro: int


def preparar_geometria(
    nodos: list[dict[str, float]],
    validos: list[bool] | None,
    poligono: list[dict[str, float]],
    ancho: int,
    alto: int,
    subset_size: int,
) -> GeometriaMalla:
    """Revalida los nodos guardados con el subset del analisis (que puede ser mayor que el de la malla)."""
    mascara = generar_mascara(ancho, alto, poligono)
    validos_subset = nodos_validos(mascara, nodos, subset_size)
    validos = [a and b for a, b in zip(validos_subset, validos or [True] * len(nodos))]
    triangulos = triangular(nodos, validos, mascara)
    if not triangulos:
        raise ValueError("Con este tamano de subset no quedan nodos validos suficientes para triangular la region")
    return GeometriaMalla(nodos=nodos, validos=validos, triangulos=triangulos, indice_centro=indice_nodo_central(nodos, validos, poligono))
