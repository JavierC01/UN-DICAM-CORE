"""Marcacion de puntos con el mouse sobre un frame (ventana de OpenCV).

Clic izquierdo: agregar punto | Clic derecho o Backspace: deshacer | Enter: aceptar | Esc: cancelar
"""

import cv2
import numpy as np

_MAX_LADO = 1000  # la ventana se escala para caber en pantalla; las coordenadas devueltas son del frame original


def marcar_puntos(
    frame: np.ndarray, titulo: str, n_puntos: int | None = None, minimo: int = 1, cerrar_poligono: bool = False
) -> list[dict[str, float]]:
    escala = min(1.0, _MAX_LADO / max(frame.shape[:2]))
    base = cv2.resize(frame, None, fx=escala, fy=escala) if escala < 1 else frame.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    puntos: list[tuple[int, int]] = []

    def on_mouse(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and (n_puntos is None or len(puntos) < n_puntos):
            puntos.append((x, y))
        elif evento == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()

    ventana = "UN-DICAM"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(ventana, on_mouse)
    requeridos = f"{n_puntos}" if n_puntos else f"minimo {minimo}"
    print(f"\n>> {titulo}\n   Marca {requeridos} puntos. Clic der./Backspace deshace, Enter acepta, Esc cancela.")

    try:
        while True:
            vista = base.copy()
            cv2.putText(vista, f"{titulo} ({len(puntos)}/{requeridos})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vista, f"{titulo} ({len(puntos)}/{requeridos})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            if len(puntos) > 1:
                cv2.polylines(vista, [np.array(puntos, dtype=np.int32)], cerrar_poligono and len(puntos) > 2, (0, 255, 0), 1)
            for i, p in enumerate(puntos):
                cv2.circle(vista, p, 4, (0, 0, 255), -1)
                cv2.putText(vista, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(ventana, vista)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in (13, 10):
                if (n_puntos and len(puntos) == n_puntos) or (not n_puntos and len(puntos) >= minimo):
                    break
                print(f"   Faltan puntos: tienes {len(puntos)}, se requieren {requeridos}.")
            elif tecla == 8 and puntos:
                puntos.pop()
            elif tecla == 27:
                raise KeyboardInterrupt("Marcacion cancelada por el usuario")
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt("Ventana cerrada por el usuario")
    finally:
        cv2.destroyWindow(ventana)
        cv2.waitKey(1)

    return [{"x": round(x / escala, 1), "y": round(y / escala, 1)} for x, y in puntos]


def marcar_rectangulo(frame: np.ndarray, titulo: str) -> dict[str, int]:
    """Region rectangular (p. ej. la pantalla de carga para OCR): 2 esquinas opuestas."""
    a, b = marcar_puntos(frame, titulo, n_puntos=2)
    x0, x1 = sorted((a["x"], b["x"]))
    y0, y1 = sorted((a["y"], b["y"]))
    return {"x": int(x0), "y": int(y0), "width": int(round(x1 - x0)), "height": int(round(y1 - y0))}


def confirmar(imagen: np.ndarray, titulo: str) -> bool:
    """Muestra una vista previa. Enter acepta (True); R repite (False); Esc cancela."""
    escala = min(1.0, _MAX_LADO / max(imagen.shape[:2]))
    vista = cv2.resize(imagen, None, fx=escala, fy=escala) if escala < 1 else imagen.copy()
    if vista.ndim == 2:
        vista = cv2.cvtColor(vista, cv2.COLOR_GRAY2BGR)
    leyenda = f"{titulo}: Enter = aceptar | R = repetir | Esc = cancelar"
    cv2.putText(vista, leyenda, (10, vista.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(vista, leyenda, (10, vista.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print("   Revisa la ventana: Enter acepta, R repite, Esc cancela.")
    ventana = "UN-DICAM"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.imshow(ventana, vista)
        while True:
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in (13, 10):
                return True
            if tecla in (ord("r"), ord("R")):
                return False
            if tecla == 27 or cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt("Cancelado por el usuario")
    finally:
        cv2.destroyWindow(ventana)
        cv2.waitKey(1)


def dibujar_malla(
    frame: np.ndarray,
    poligono: list[dict[str, float]],
    nodos: list[dict[str, float]],
    validos: list[bool],
    triangulos: list[tuple[int, int, int]],
) -> np.ndarray:
    """Poligono de la probeta (amarillo), triangulos (verde), nodos validos (verde) e invalidos (rojo)."""
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    g = max(1, int(max(img.shape[:2]) / 700))
    pts = lambda idx: np.array([[int(nodos[i]["x"]), int(nodos[i]["y"])] for i in idx], dtype=np.int32)
    cv2.polylines(img, [np.array([[int(p["x"]), int(p["y"])] for p in poligono], dtype=np.int32)], True, (0, 255, 255), g)
    for tri in triangulos:
        cv2.polylines(img, [pts(tri)], True, (0, 200, 0), g)
    for n, v in zip(nodos, validos):
        cv2.circle(img, (int(n["x"]), int(n["y"])), 3 * g, (0, 220, 0) if v else (0, 0, 255), -1)
    return img
