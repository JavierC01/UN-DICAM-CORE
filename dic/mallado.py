"""Identificacion de la probeta sobre el frame de referencia y generacion de la malla de analisis."""

import numpy as np

from dic.consola import pedir_int
from dic.interactivo import confirmar, dibujar_malla, marcar_puntos
from dic.rastreo import GeometriaMalla, area_poligono, construir_malla, preparar_geometria


def identificar_probeta(frame: np.ndarray, subset_size: int) -> tuple[dict, GeometriaMalla]:
    """El usuario contornea la probeta, elige la densidad de la malla y confirma la vista previa.

    Devuelve ({"poligono_px", "filas", "columnas", "subset_size"}, geometria de la malla aceptada).
    """
    alto, ancho = frame.shape[:2]
    poligono = None
    filas, columnas = 5, 5
    while True:
        if poligono is None:
            print("\n   Contornea la zona de la probeta a analizar (minimo 3 puntos, en orden alrededor del borde).")
            poligono = marcar_puntos(frame, "Probeta: contorno de la zona de analisis", minimo=3, cerrar_poligono=True)
            if area_poligono(poligono) < ancho * alto * 0.005:
                print("   La zona marcada es demasiado pequena; marca un contorno mas amplio.")
                poligono = None
                continue
        print("   Densidad de la malla (nodos en la zona marcada):")
        filas = pedir_int("Filas de nodos", filas, minimo=2, maximo=60)
        columnas = pedir_int("Columnas de nodos", columnas, minimo=2, maximo=60)
        try:
            malla = construir_malla(ancho, alto, poligono, filas, columnas, subset_size)
            geometria = preparar_geometria(malla.nodos, malla.validos, poligono, ancho, alto, subset_size)
        except ValueError as exc:
            print(f"   No se pudo generar la malla: {exc}")
            poligono = None
            continue
        print(
            f"   Malla: {geometria.validos.count(True)}/{len(geometria.nodos)} nodos validos, "
            f"{len(geometria.triangulos)} triangulos (subset {subset_size} px)"
        )
        if confirmar(dibujar_malla(frame, poligono, geometria.nodos, geometria.validos, geometria.triangulos), "Malla"):
            return {"poligono_px": poligono, "filas": filas, "columnas": columnas, "subset_size": subset_size}, geometria
        print("   Repetir: se vuelve a marcar el contorno.")
        poligono = None
