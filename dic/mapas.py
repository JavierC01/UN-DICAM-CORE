from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


def guardar_mapa(
    imagen_ref: np.ndarray,
    nodos_px: list[dict[str, float]],
    triangulos: list[tuple[int, int, int]],
    valores: list[float],
    titulo: str,
    ruta: Path,
) -> None:
    """Dibuja la deformacion por triangulo sobre el frame de referencia; los triangulos NaN quedan transparentes."""
    alto, ancho = imagen_ref.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6 * alto / ancho))
    ax.imshow(imagen_ref, cmap="gray", extent=(0, ancho, alto, 0))

    valores_arr = np.asarray(valores, dtype=np.float64)
    finitos = np.isfinite(valores_arr)
    if finitos.any():
        tri = Triangulation(
            [n["x"] for n in nodos_px], [n["y"] for n in nodos_px], triangles=np.asarray(triangulos), mask=~finitos
        )
        capa = ax.tripcolor(tri, facecolors=np.where(finitos, valores_arr, 0.0), cmap="jet", alpha=0.6)
        fig.colorbar(capa, ax=ax, fraction=0.046)
    else:
        ax.text(ancho / 2, alto / 2, "Sin datos validos", color="red", ha="center")

    ax.set_xlim(0, ancho)
    ax.set_ylim(alto, 0)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ruta, format="jpg", dpi=120)
    plt.close(fig)
