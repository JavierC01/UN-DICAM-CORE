import numpy as np
from numba import njit, prange


@njit(cache=True)
def zncc(subset: np.ndarray, ventana: np.ndarray) -> float:
    a = subset.astype(np.float64) - subset.mean()
    b = ventana.astype(np.float64) - ventana.mean()
    denominador = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominador == 0:
        return 0.0
    return float(np.sum(a * b) / denominador)


@njit(cache=True)
def extraer_subset(imagen: np.ndarray, centro: tuple[int, int], subset_size: int) -> np.ndarray:
    half = subset_size // 2
    cx, cy = centro
    return imagen[cy - half : cy - half + subset_size, cx - half : cx - half + subset_size]


@njit(cache=True)
def centro_valido(imagen: np.ndarray, centro: tuple[int, int], subset_size: int) -> bool:
    half = subset_size // 2
    cx, cy = centro
    alto, ancho = imagen.shape[0], imagen.shape[1]
    return cy - half >= 0 and cx - half >= 0 and cy - half + subset_size <= alto and cx - half + subset_size <= ancho


@njit(cache=True, parallel=True)
def mapa_correlacion(
    imagen_ref: np.ndarray,
    imagen_def: np.ndarray,
    centro: tuple[int, int],
    subset_size: int,
    search_range: int,
) -> np.ndarray:
    if not centro_valido(imagen_ref, centro, subset_size):
        raise ValueError("El subset de referencia excede los limites de la imagen")

    subset = extraer_subset(imagen_ref, centro, subset_size)
    cx, cy = centro
    lado = 2 * search_range + 1
    mapa = np.full((lado, lado), -np.inf, dtype=np.float64)

    for j in prange(lado):
        dy = j - search_range
        for i in range(lado):
            dx = i - search_range
            candidato = (cx + dx, cy + dy)
            if not centro_valido(imagen_def, candidato, subset_size):
                continue
            ventana = extraer_subset(imagen_def, candidato, subset_size)
            mapa[j, i] = zncc(subset, ventana)
    return mapa


def pico_entero(mapa: np.ndarray) -> tuple[int, int, float]:
    if not np.any(np.isfinite(mapa)):
        raise ValueError("El mapa de correlacion no tiene candidatos validos dentro de los limites de la imagen")
    idx = int(np.argmax(mapa))
    j, i = np.unravel_index(idx, mapa.shape)
    return int(i), int(j), float(mapa[j, i])
