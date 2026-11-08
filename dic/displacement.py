from dataclasses import dataclass

import numpy as np

from dic.correlation import mapa_correlacion, pico_entero


def lsq_refinement(mapa: np.ndarray, pico: tuple[int, int]) -> tuple[float, float]:
    i, j = pico
    alto, ancho = mapa.shape
    if i <= 0 or j <= 0 or i >= ancho - 1 or j >= alto - 1:
        return 0.0, 0.0

    vecinos = mapa[j - 1 : j + 2, i - 1 : i + 2]
    if not np.all(np.isfinite(vecinos)):
        return 0.0, 0.0

    x, y = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    x, y, z = x.ravel(), y.ravel(), vecinos.astype(np.float64).ravel()

    diseno = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    a, b, c, d, e, _ = np.linalg.lstsq(diseno, z, rcond=None)[0]

    hessiano = np.array([[2 * a, c], [c, 2 * b]])
    if abs(np.linalg.det(hessiano)) < 1e-9:
        return 0.0, 0.0

    delta_x, delta_y = np.linalg.solve(hessiano, [-d, -e])
    if abs(delta_x) > 1 or abs(delta_y) > 1:
        return 0.0, 0.0
    return float(delta_x), float(delta_y)


@dataclass(frozen=True)
class ResultadoNodo:
    u_px: float
    v_px: float
    coef_correlacion: float


def calcular_desplazamiento_nodo(
    imagen_ref: np.ndarray,
    imagen_def: np.ndarray,
    centro: tuple[int, int],
    subset_size: int,
    search_range: int,
) -> ResultadoNodo:
    mapa = mapa_correlacion(imagen_ref, imagen_def, centro, subset_size, search_range)
    i, j, coef = pico_entero(mapa)
    delta_x, delta_y = lsq_refinement(mapa, (i, j))

    return ResultadoNodo(
        u_px=(i - search_range) + delta_x,
        v_px=(j - search_range) + delta_y,
        coef_correlacion=coef,
    )


def rastrear_malla(
    imagen_ref: np.ndarray,
    imagen_def: np.ndarray,
    nodos: list[dict[str, float]],
    subset_size: int,
    search_range: int,
    validos: list[bool] | None = None,
) -> list[ResultadoNodo | None]:
    """Rastrea cada nodo valido; los invalidos y los que no se pudieron rastrear en este frame quedan en None."""
    resultados: list[ResultadoNodo | None] = []
    for i, nodo in enumerate(nodos):
        if validos is not None and not validos[i]:
            resultados.append(None)
            continue
        try:
            resultados.append(
                calcular_desplazamiento_nodo(
                    imagen_ref, imagen_def, (int(round(nodo["x"])), int(round(nodo["y"]))), subset_size, search_range
                )
            )
        except ValueError:
            resultados.append(None)
    return resultados
