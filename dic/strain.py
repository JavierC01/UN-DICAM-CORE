import math
from dataclasses import dataclass

import numpy as np

# La malla de rastreo es una grilla filas x columnas ordenada por filas (ver rastreo.construir_malla), p. ej. 3x3:
#   0 1 2
#   3 4 5
#   6 7 8
# Los nodos invalidos (fuera de la forma) o que no se pudieron rastrear tienen desplazamiento None.

Desplazamiento = tuple[float, float] | None


@dataclass(frozen=True)
class CampoDeformacion:
    eps_xx: list[list[float]]
    eps_yy: list[list[float]]
    eps_xy: list[list[float]]
    eps_xx_medio: float
    eps_yy_medio: float
    eps_xy_medio: float


@dataclass(frozen=True)
class CampoTriangular:
    """Deformacion constante por triangulo (elementos CST). Los triangulos con un nodo perdido valen NaN."""

    triangulos: list[tuple[int, int, int]]
    eps_xx: list[float]
    eps_yy: list[float]
    eps_xy: list[float]
    eps_xx_medio: float
    eps_yy_medio: float
    eps_xy_medio: float


def _media(valores: list[float], pesos: list[float] | None = None) -> float:
    v = np.asarray(valores, dtype=np.float64)
    w = np.ones_like(v) if pesos is None else np.asarray(pesos, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any() or w[ok].sum() == 0:
        return math.nan
    return float(np.average(v[ok], weights=w[ok]))


def _deformacion_cuadrante(
    esquinas: tuple[int, int, int, int],
    nodos_mm: list[dict[str, float]],
    desplazamientos_mm: list[Desplazamiento],
) -> tuple[float, float, float]:
    if any(desplazamientos_mm[i] is None for i in esquinas):
        return math.nan, math.nan, math.nan
    sup_izq, sup_der, inf_izq, inf_der = esquinas

    x_izq = (nodos_mm[sup_izq]["x"] + nodos_mm[inf_izq]["x"]) / 2
    x_der = (nodos_mm[sup_der]["x"] + nodos_mm[inf_der]["x"]) / 2
    y_sup = (nodos_mm[sup_izq]["y"] + nodos_mm[sup_der]["y"]) / 2
    y_inf = (nodos_mm[inf_izq]["y"] + nodos_mm[inf_der]["y"]) / 2

    dx = x_der - x_izq
    dy = y_inf - y_sup
    if dx == 0 or dy == 0:
        return math.nan, math.nan, math.nan

    u_izq = (desplazamientos_mm[sup_izq][0] + desplazamientos_mm[inf_izq][0]) / 2
    u_der = (desplazamientos_mm[sup_der][0] + desplazamientos_mm[inf_der][0]) / 2
    u_sup = (desplazamientos_mm[sup_izq][0] + desplazamientos_mm[sup_der][0]) / 2
    u_inf = (desplazamientos_mm[inf_izq][0] + desplazamientos_mm[inf_der][0]) / 2

    v_izq = (desplazamientos_mm[sup_izq][1] + desplazamientos_mm[inf_izq][1]) / 2
    v_der = (desplazamientos_mm[sup_der][1] + desplazamientos_mm[inf_der][1]) / 2
    v_sup = (desplazamientos_mm[sup_izq][1] + desplazamientos_mm[sup_der][1]) / 2
    v_inf = (desplazamientos_mm[inf_izq][1] + desplazamientos_mm[inf_der][1]) / 2

    eps_xx = (u_der - u_izq) / dx
    eps_yy = (v_inf - v_sup) / dy
    eps_xy = 0.5 * ((u_inf - u_sup) / dy + (v_der - v_izq) / dx)
    return eps_xx, eps_yy, eps_xy


def calcular_deformaciones(
    nodos_mm: list[dict[str, float]], desplazamientos_mm: list[Desplazamiento], columnas: int = 3
) -> CampoDeformacion:
    """Calcula la deformacion ingenieril (exx, eyy, exy) en cada celda de la grilla de nodos

    (filas-1 x columnas-1 celdas), usando diferencias finitas entre los 4 nodos de cada celda.
    Una celda con alguna esquina invalida (desplazamiento None) vale NaN.
    """
    filas = len(nodos_mm) // columnas if columnas > 0 else 0
    if filas < 2 or columnas < 2 or len(nodos_mm) != filas * columnas or len(desplazamientos_mm) != len(nodos_mm):
        raise ValueError("Se requiere una grilla de nodos de al menos 2x2 organizada por filas")

    valores = []
    for f in range(filas - 1):
        fila = []
        for c in range(columnas - 1):
            sup_izq = f * columnas + c
            inf_izq = sup_izq + columnas
            esquinas = (sup_izq, sup_izq + 1, inf_izq, inf_izq + 1)
            fila.append(_deformacion_cuadrante(esquinas, nodos_mm, desplazamientos_mm))
        valores.append(fila)
    celdas = [v for fila in valores for v in fila]

    return CampoDeformacion(
        eps_xx=[[v[0] for v in fila] for fila in valores],
        eps_yy=[[v[1] for v in fila] for fila in valores],
        eps_xy=[[v[2] for v in fila] for fila in valores],
        eps_xx_medio=_media([v[0] for v in celdas]),
        eps_yy_medio=_media([v[1] for v in celdas]),
        eps_xy_medio=_media([v[2] for v in celdas]),
    )


def _gradiente_triangulo(p: np.ndarray, valores: np.ndarray) -> tuple[float, float]:
    """Gradiente (d/dx, d/dy) del campo lineal que pasa por los 3 vertices p (3x2)."""
    matriz = np.column_stack([np.ones(3), p])
    _, gx, gy = np.linalg.solve(matriz, valores)
    return float(gx), float(gy)


def deformacion_triangulos(
    nodos_mm: list[dict[str, float]], desplazamientos_mm: list[Desplazamiento], triangulos: list[tuple[int, int, int]]
) -> CampoTriangular:
    """Elementos de deformacion constante: u y v son lineales en cada triangulo.

    Las medias se ponderan por el area de cada triangulo, asi una malla irregular no sesga el promedio.
    """
    if not triangulos:
        raise ValueError("Se requiere al menos un triangulo para calcular deformaciones")
    exx, eyy, exy, areas = [], [], [], []
    for tri in triangulos:
        p = np.array([[nodos_mm[i]["x"], nodos_mm[i]["y"]] for i in tri], dtype=np.float64)
        a, b = p[1] - p[0], p[2] - p[0]
        areas.append(0.5 * abs(a[0] * b[1] - a[1] * b[0]))
        if any(desplazamientos_mm[i] is None for i in tri):
            exx.append(math.nan)
            eyy.append(math.nan)
            exy.append(math.nan)
            continue
        u = np.array([desplazamientos_mm[i][0] for i in tri])
        v = np.array([desplazamientos_mm[i][1] for i in tri])
        du_dx, du_dy = _gradiente_triangulo(p, u)
        dv_dx, dv_dy = _gradiente_triangulo(p, v)
        exx.append(du_dx)
        eyy.append(dv_dy)
        exy.append(0.5 * (du_dy + dv_dx))
    return CampoTriangular(
        triangulos=list(triangulos),
        eps_xx=exx,
        eps_yy=eyy,
        eps_xy=exy,
        eps_xx_medio=_media(exx, areas),
        eps_yy_medio=_media(eyy, areas),
        eps_xy_medio=_media(exy, areas),
    )


def localizar_punto(
    punto: dict[str, float], nodos: list[dict[str, float]], triangulos: list[tuple[int, int, int]]
) -> tuple[tuple[int, int, int], tuple[float, float, float]] | None:
    """Triangulo que contiene el punto y sus coordenadas baricentricas, o None si cae fuera de la malla."""
    x, y = punto["x"], punto["y"]
    for tri in triangulos:
        (x1, y1), (x2, y2), (x3, y3) = [(nodos[i]["x"], nodos[i]["y"]) for i in tri]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if det == 0:
            continue
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        l3 = 1 - l1 - l2
        if min(l1, l2, l3) >= -1e-9:
            return tri, (l1, l2, l3)
    return None


def interpolar_desplazamiento(
    ubicacion: tuple[tuple[int, int, int], tuple[float, float, float]] | None,
    desplazamientos: list[Desplazamiento],
) -> tuple[float, float]:
    """Desplazamiento lineal (consistente con CST) en un punto ya localizado; NaN si falta algun vertice."""
    if ubicacion is None:
        return math.nan, math.nan
    tri, pesos = ubicacion
    if any(desplazamientos[i] is None for i in tri):
        return math.nan, math.nan
    u = sum(w * desplazamientos[i][0] for i, w in zip(tri, pesos))
    v = sum(w * desplazamientos[i][1] for i, w in zip(tri, pesos))
    return float(u), float(v)
