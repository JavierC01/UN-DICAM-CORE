"""Fuentes de imagenes: deteccion de camaras, informacion del video y descomposicion en frames a N datos/s."""

import hashlib
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# (indice del frame en la fuente, tiempo desde la referencia en s, imagen BGR)
Muestra = tuple[int, float, np.ndarray]
MAX_CAMARAS = 5


class FuenteError(RuntimeError):
    pass


def _api_camara() -> int:
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


@dataclass(frozen=True)
class InfoFuente:
    ancho: int
    alto: int
    fps: float
    n_frames: int  # 0 en camaras

    @property
    def duracion_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 and self.n_frames > 0 else 0.0


# ----------------------------------------------------------------------------- camaras

def abrir_camara(indice: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(indice, _api_camara())
    if not cap.isOpened():
        cap.release()
        raise FuenteError(f"No se pudo abrir la camara {indice}")
    return cap


def listar_camaras(max_dispositivos: int = MAX_CAMARAS) -> list[tuple[int, InfoFuente]]:
    """Camaras conectadas que entregan imagen. Los indices son consecutivos: se para en el primero vacio."""
    camaras = []
    for indice in range(max_dispositivos):
        cap = cv2.VideoCapture(indice, _api_camara())
        try:
            if not (cap.isOpened() and cap.read()[0]):
                break
            info = InfoFuente(
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 0,
            )
            camaras.append((indice, info))
        finally:
            cap.release()
    return camaras


def frame_camara(indice: int, descartar: int = 10) -> np.ndarray:
    """Un frame estable de la camara (los primeros suelen salir oscuros mientras ajusta la exposicion)."""
    cap = abrir_camara(indice)
    try:
        frame = None
        for _ in range(descartar + 1):
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        cap.release()
    if frame is None:
        raise FuenteError(f"La camara {indice} no entrega imagen")
    return frame


def capturar_camara(
    indice: int,
    datos_por_segundo: float,
    duracion_s: float,
    progreso: Callable[[int, int], None] | None = None,
    vista_previa: bool = True,
) -> list[Muestra]:
    """Captura a `datos_por_segundo` durante `duracion_s`. El tiempo de cada muestra es el reloj real de captura.

    Primero se captura todo y despues se procesa, para que el DIC no frene el ritmo de muestreo.
    Esc en la vista previa termina la captura antes de tiempo.
    """
    total = max(2, int(round(datos_por_segundo * duracion_s)) + 1)
    periodo = 1.0 / datos_por_segundo
    cap = abrir_camara(indice)
    muestras: list[Muestra] = []
    try:
        for _ in range(5):
            cap.read()
        t0 = time.perf_counter()
        while len(muestras) < total:
            if not cap.grab():
                raise FuenteError("La camara dejo de entregar imagen durante la captura")
            t = time.perf_counter() - t0
            if t + 1e-4 < len(muestras) * periodo:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            muestras.append((len(muestras), t, frame))
            if progreso:
                progreso(len(muestras), total)
            if vista_previa:
                vista = frame.copy()
                cv2.putText(vista, f"Capturando {len(muestras)}/{total} - Esc termina", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("UN-DICAM captura", vista)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if vista_previa:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    if len(muestras) < 2:
        raise FuenteError("Se necesitan al menos 2 imagenes capturadas")
    return muestras


# ----------------------------------------------------------------------------- videos

def abrir_video(ruta: str | Path) -> cv2.VideoCapture:
    if not Path(ruta).is_file():
        raise FuenteError(f"El video no existe: {ruta}")
    cap = cv2.VideoCapture(str(ruta), cv2.CAP_ANY)
    if not cap.isOpened():
        raise FuenteError(f"No se pudo abrir el video: {ruta}")
    return cap


def info_video(ruta: str | Path) -> InfoFuente:
    cap = abrir_video(ruta)
    try:
        return InfoFuente(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS) or 0.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        cap.release()


def frame_video(ruta: str | Path, indice: int = 0) -> np.ndarray:
    cap = abrir_video(ruta)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise FuenteError(f"No se pudo leer el frame {indice} del video")
    return frame


def paso_frames(fps: float, datos_por_segundo: float) -> int:
    """Cada cuantos frames del video se toma un dato (1 = todos)."""
    return max(1, int(round(fps / datos_por_segundo)))


def contar_muestras(frame_inicio: int, frame_fin: int, paso: int) -> int:
    """Imagenes que se analizan ademas de la referencia."""
    return (frame_fin - frame_inicio) // paso


def descomponer_video(ruta: str | Path, frame_inicio: int, frame_fin: int, paso: int, fps: float) -> Iterator[Muestra]:
    """Referencia en frame_inicio y despues 1 de cada `paso` frames hasta frame_fin (inclusive).

    Los frames intermedios solo se avanzan (grab), sin decodificarlos.
    """
    if fps <= 0:
        raise FuenteError("El video no informa fps; no se puede asignar tiempo a los frames")
    cap = abrir_video(ruta)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
        ok, frame = cap.read()
        if not ok:
            raise FuenteError("No fue posible leer el frame de referencia")
        yield frame_inicio, 0.0, frame
        for indice in range(frame_inicio + 1, frame_fin + 1):
            if (indice - frame_inicio) % paso:
                if not cap.grab():
                    return
                continue
            ok, frame = cap.read()
            if not ok:
                return
            yield indice, (indice - frame_inicio) / fps, frame
    finally:
        cap.release()


def sha256_archivo(ruta: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            sha.update(bloque)
    return sha.hexdigest()


def memoria_estimada_mb(info: InfoFuente, n_imagenes: int) -> float:
    return info.ancho * info.alto * 3 * n_imagenes / 1e6

