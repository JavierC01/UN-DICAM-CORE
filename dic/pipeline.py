import math
from dataclasses import dataclass

import numpy as np

from dic.displacement import ResultadoNodo, rastrear_malla
from dic.rastreo import GeometriaMalla
from dic.strain import CampoTriangular, Desplazamiento, deformacion_triangulos


@dataclass(frozen=True)
class ResultadoFrame:
    nodos: list[ResultadoNodo | None]
    desplazamientos_mm: list[Desplazamiento]
    coef_correlacion_medio: float
    campo: CampoTriangular
    nodo_centro: ResultadoNodo | None
    u_mm_centro: float
    v_mm_centro: float
    nodos_perdidos: int


def procesar_par_frames(
    imagen_ref: np.ndarray,
    imagen_def: np.ndarray,
    geometria: GeometriaMalla,
    mm_por_px: float,
    subset_size: int,
    search_range: int,
) -> ResultadoFrame:
    """Rastrea los nodos validos entre dos frames y calcula la deformacion por triangulos (CST)."""
    resultados = rastrear_malla(imagen_ref, imagen_def, geometria.nodos, subset_size, search_range, geometria.validos)
    rastreados = [r for r in resultados if r is not None]
    if not rastreados:
        raise ValueError("Ningun nodo se pudo rastrear en este frame")

    nodos_mm = [{"x": n["x"] * mm_por_px, "y": n["y"] * mm_por_px} for n in geometria.nodos]
    desplazamientos_mm = [None if r is None else (r.u_px * mm_por_px, r.v_px * mm_por_px) for r in resultados]
    campo = deformacion_triangulos(nodos_mm, desplazamientos_mm, geometria.triangulos)

    centro = resultados[geometria.indice_centro]
    return ResultadoFrame(
        nodos=resultados,
        desplazamientos_mm=desplazamientos_mm,
        coef_correlacion_medio=float(np.mean([r.coef_correlacion for r in rastreados])),
        campo=campo,
        nodo_centro=centro,
        u_mm_centro=centro.u_px * mm_por_px if centro else math.nan,
        v_mm_centro=centro.v_px * mm_por_px if centro else math.nan,
        nodos_perdidos=sum(1 for r, v in zip(resultados, geometria.validos) if v and r is None),
    )
