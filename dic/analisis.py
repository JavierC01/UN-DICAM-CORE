"""Analisis DIC completo: rastreo imagen a imagen, deformaciones, carga, ensayo y metadatos.

Version independiente (sin base de datos ni Celery) de la tarea process_full_analysis original.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dic.cargas.sincronizacion import cargas_en_frames
from dic.ensayos import ajuste_lineal, obtener_ensayo
from dic.fuentes import (
    FuenteError,
    capturar_camara,
    contar_muestras,
    descomponer_video,
    info_video,
    paso_frames,
    sha256_archivo,
)
from dic.mapas import guardar_mapa
from dic.ocr import extract_load_value
from dic.pipeline import procesar_par_frames
from dic.probeta import MedidasProbeta, Probeta
from dic.rastreo import construir_malla, preparar_geometria
from dic.strain import interpolar_desplazamiento, localizar_punto

VERSION = "1.1.0"

Progreso = Callable[[int, int], None]


class AnalisisError(RuntimeError):
    pass


@dataclass
class SerieCarga:
    """Serie de la maquina ya convertida a segundos y newton (ver dic.cargas.importador)."""

    tiempos_s: list[float]
    cargas_n: list[float]
    nombre_archivo: str
    sha256: str
    mapeo: dict
    sincronizacion: dict = field(default_factory=dict)


@dataclass
class ConfigAnalisis:
    fuente: int | str  # indice de camara (int) o ruta de video (str)
    medidas: MedidasProbeta
    poligono: list[dict[str, float]]
    datos_por_segundo: float | None = None  # None: todos los frames del video
    frame_inicio: int = 0  # solo video
    frame_fin: int | None = None  # solo video; None = ultimo frame
    duracion_s: float = 10.0  # solo camara
    filas: int = 3
    columnas: int = 3
    subset_malla: int = 32
    subset_size: int = 32
    search_range: int = 20
    tipo_ensayo: str = "traccion"
    parametros_ensayo: dict = field(default_factory=dict)
    puntos_ensayo: list[dict[str, float]] | None = None
    ajuste: dict | None = None
    fuente_carga: str = "ninguna"  # ninguna | ocr | archivo
    roi_carga: dict | None = None
    serie_carga: SerieCarga | None = None


def es_camara(fuente: int | str) -> bool:
    return isinstance(fuente, int)


def sin_nan(valor):
    """JSON no admite NaN: los datos faltantes se guardan como null."""
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, dict):
        return {k: sin_nan(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [sin_nan(v) for v in valor]
    return valor


def _max_finito(valores: list[float]) -> float:
    finitos = [v for v in valores if math.isfinite(v)]
    return max(finitos) if finitos else math.nan


def _paso_px(nodos: list[dict], columnas: int) -> dict:
    return {
        "x": nodos[1]["x"] - nodos[0]["x"] if columnas > 1 else None,
        "y": nodos[columnas]["y"] - nodos[0]["y"] if len(nodos) > columnas else None,
    }


def _preparar_muestras(cfg: ConfigAnalisis, progreso_captura: Progreso | None):
    """Devuelve (iterador de muestras, total sin la referencia, info para metadatos)."""
    if es_camara(cfg.fuente):
        if not cfg.datos_por_segundo:
            raise AnalisisError("Indica los datos por segundo para capturar desde la camara")
        muestras = capturar_camara(cfg.fuente, cfg.datos_por_segundo, cfg.duracion_s, progreso_captura)
        duracion_real = muestras[-1][1]
        info = {
            "tipo": "camara",
            "indice_camara": cfg.fuente,
            "sha256": None,
            "datos_por_segundo_pedidos": cfg.datos_por_segundo,
            "datos_por_segundo_reales": (len(muestras) - 1) / duracion_real if duracion_real > 0 else math.nan,
            "duracion_s": duracion_real,
            "imagenes": len(muestras),
        }
        return iter(muestras), len(muestras) - 1, info

    video = info_video(cfg.fuente)
    if video.fps <= 0:
        raise AnalisisError("El video no informa fps; no se puede asignar tiempo a los frames")
    ultimo = video.n_frames - 1 if video.n_frames > 0 else None
    frame_fin = cfg.frame_fin if cfg.frame_fin is not None else ultimo
    if frame_fin is None:
        raise AnalisisError("No se pudo determinar el numero de frames del video; indica el frame final")
    if ultimo is not None:
        frame_fin = min(frame_fin, ultimo)
    if frame_fin <= cfg.frame_inicio:
        raise AnalisisError("El frame final debe ser posterior al de referencia")
    paso = paso_frames(video.fps, cfg.datos_por_segundo or video.fps)
    total = contar_muestras(cfg.frame_inicio, frame_fin, paso)
    if total < 1:
        raise AnalisisError("Con esos datos por segundo no queda ninguna imagen despues de la referencia; amplia el rango")
    info = {
        "tipo": "archivo",
        "ruta": str(cfg.fuente),
        "sha256": sha256_archivo(cfg.fuente),
        "fps_video": video.fps,
        "datos_por_segundo_pedidos": cfg.datos_por_segundo or video.fps,
        "datos_por_segundo_reales": video.fps / paso,
        "paso_frames": paso,
        "frame_inicio": cfg.frame_inicio,
        "frame_fin": frame_fin,
        "imagenes": total + 1,
    }
    return descomponer_video(cfg.fuente, cfg.frame_inicio, frame_fin, paso, video.fps), total, info


def ejecutar_analisis(
    cfg: ConfigAnalisis,
    output_dir: Path,
    progreso: Progreso | None = None,
    progreso_captura: Progreso | None = None,
) -> dict:
    """Ejecuta el analisis completo, guarda los mapas en output_dir y devuelve el diccionario de resultados."""
    advertencias: list[str] = []
    probeta = Probeta(cfg.medidas)
    ensayo = obtener_ensayo(cfg.tipo_ensayo)
    try:
        ensayo.validar_parametros(cfg.parametros_ensayo)
    except ValueError as exc:
        raise AnalisisError(str(exc)) from exc
    if len(cfg.poligono) < 3:
        raise AnalisisError("El poligono de la region de interes necesita al menos 3 puntos")
    if cfg.fuente_carga == "archivo" and cfg.serie_carga is None:
        raise AnalisisError("La fuente de carga es 'archivo' pero no se cargo ningun archivo de la maquina")
    if cfg.fuente_carga == "ocr" and not cfg.roi_carga:
        raise AnalisisError("La fuente de carga es 'ocr' pero no hay roi de la pantalla de carga")
    if cfg.fuente_carga == "ninguna":
        advertencias.append("Sin fuente de carga configurada: carga y esfuerzo quedan nulos")

    try:
        muestras, total, info_fuente = _preparar_muestras(cfg, progreso_captura)
        _, _, ref_frame = next(muestras)
    except FuenteError as exc:
        raise AnalisisError(str(exc)) from exc
    except StopIteration as exc:
        raise AnalisisError("No fue posible leer la imagen de referencia") from exc
    imagen_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    alto, ancho = imagen_ref.shape

    # Malla de rastreo sobre el poligono marcado
    try:
        malla = construir_malla(ancho, alto, cfg.poligono, cfg.filas, cfg.columnas, cfg.subset_malla)
        geometria = preparar_geometria(malla.nodos, malla.validos, cfg.poligono, ancho, alto, cfg.subset_size)
    except ValueError as exc:
        raise AnalisisError(str(exc)) from exc
    invalidos = geometria.validos.count(False)
    if invalidos:
        advertencias.append(
            f"{invalidos} de {len(geometria.nodos)} nodos quedan fuera de la region para un subset de "
            f"{cfg.subset_size} px y no se rastrean"
        )

    # Extensometro virtual del ensayo
    ubicaciones = None
    puntos_ensayo_mm = []
    if cfg.puntos_ensayo and len(cfg.puntos_ensayo) == ensayo.n_puntos:
        ubicaciones = [localizar_punto(p, geometria.nodos, geometria.triangulos) for p in cfg.puntos_ensayo]
        if any(u is None for u in ubicaciones):
            advertencias.append("Algun punto del extensometro virtual quedo fuera de la malla valida; su medida es nula")
        puntos_ensayo_mm = [(p["x"] * probeta.mm_por_px, p["y"] * probeta.mm_por_px) for p in cfg.puntos_ensayo]
    else:
        advertencias.append(
            f"No se marcaron los {ensayo.n_puntos} puntos del ensayo de {ensayo.nombre} ({ensayo.puntos}); "
            "no hay curva del ensayo ni propiedad"
        )

    frames: list[dict] = []
    fallos_ocr = 0
    no_rastreados = 0
    ultimo_campo = None
    try:
        for k, (indice, tiempo_s, frame) in enumerate(muestras, 1):
            if progreso:
                progreso(k, total)
            imagen_def = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                resultado = procesar_par_frames(
                    imagen_ref, imagen_def, geometria, probeta.mm_por_px, cfg.subset_size, cfg.search_range
                )
            except ValueError:
                no_rastreados += 1
                continue
            ultimo_campo = resultado.campo

            carga_n = math.nan
            if cfg.fuente_carga == "ocr":
                try:
                    carga_n = extract_load_value(frame, cfg.roi_carga)
                except Exception:
                    fallos_ocr += 1  # se descarta el punto; nunca se reemplaza por 0

            medida = math.nan
            if ubicaciones is not None:
                desplaz = [interpolar_desplazamiento(u, resultado.desplazamientos_mm) for u in ubicaciones]
                medida = ensayo.medir(puntos_ensayo_mm, desplaz, cfg.parametros_ensayo)

            frames.append(
                {
                    "frame": indice,
                    "tiempo_s": tiempo_s,
                    "u_mm": resultado.u_mm_centro,
                    "v_mm": resultado.v_mm_centro,
                    "eps_xx": resultado.campo.eps_xx_medio,
                    "eps_yy": resultado.campo.eps_yy_medio,
                    "eps_xy": resultado.campo.eps_xy_medio,
                    "carga_n": carga_n,
                    "medida_ensayo": medida,
                    "coef_correlacion": resultado.coef_correlacion_medio,
                    "nodos_perdidos": resultado.nodos_perdidos,
                }
            )
    except FuenteError as exc:
        raise AnalisisError(str(exc)) from exc
    if not frames:
        raise AnalisisError("No fue posible rastrear ninguna imagen; revisa el subset y el rango de busqueda")
    if no_rastreados:
        advertencias.append(f"{no_rastreados} imagenes no se pudieron rastrear y se omitieron")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carga
    carga_info: dict = {"fuente": cfg.fuente_carga}
    serie = cfg.serie_carga
    if cfg.fuente_carga == "ocr":
        carga_info["fallos_ocr"] = fallos_ocr
        if fallos_ocr:
            advertencias.append(f"El OCR no pudo leer la carga en {fallos_ocr} de {len(frames)} imagenes; esos puntos se descartan")
    elif cfg.fuente_carga == "archivo":
        sinc = serie.sincronizacion
        try:
            cargas, interpolados, descartados = cargas_en_frames(
                [f["tiempo_s"] for f in frames],
                serie.tiempos_s,
                serie.cargas_n,
                offset_s=float(sinc.get("offset_s", 0.0)),
                evento_inicio=sinc.get("evento_inicio", "ninguno"),
                umbral_n=sinc.get("umbral_n"),
            )
        except ValueError as exc:
            raise AnalisisError(f"No se pudo sincronizar la carga: {exc}") from exc
        for f, c in zip(frames, cargas):
            f["carga_n"] = c
        carga_info.update(interpolados=interpolados, descartados=descartados)
        if descartados:
            advertencias.append(f"{descartados} imagenes caen fuera del registro de la maquina; su carga queda nula")

    for f in frames:
        f["esfuerzo_mpa"] = probeta.calcular_esfuerzo(f["carga_n"])
        f["respuesta_ensayo"] = ensayo.calcular_respuesta(f["carga_n"], probeta.area_mm2, cfg.parametros_ensayo)
    carga_info["frames_con_carga"] = sum(1 for f in frames if math.isfinite(f["carga_n"]))

    # Mapas de deformacion de la ultima imagen
    for componente in ("xx", "yy", "xy"):
        guardar_mapa(
            imagen_ref,
            geometria.nodos,
            ultimo_campo.triangulos,
            getattr(ultimo_campo, f"eps_{componente}"),
            f"Deformacion {componente.upper()} (ultima imagen)",
            output_dir / f"mapa_{componente}.jpg",
        )

    columna = lambda clave: [f[clave] for f in frames]
    datos_ensayo = None
    if ubicaciones is not None:
        datos_ensayo = {
            **ensayo.describir(),
            "parametros": cfg.parametros_ensayo,
            "curva": {"x": columna("medida_ensayo"), "y": columna("respuesta_ensayo")},
            "ajuste": None,
        }
        if cfg.ajuste:
            try:
                datos_ensayo["ajuste"] = ajuste_lineal(columna("medida_ensayo"), columna("respuesta_ensayo"), **cfg.ajuste)
            except ValueError as exc:
                advertencias.append(f"No se pudo calcular el ajuste lineal: {exc}")

    pedidos, reales = info_fuente["datos_por_segundo_pedidos"], info_fuente["datos_por_segundo_reales"]
    if math.isfinite(reales) and reales < 0.9 * pedidos:
        advertencias.append(
            f"La camara solo entrego {reales:.3g} datos/s de los {pedidos:g} pedidos; los tiempos registrados son los reales"
        )
    if info_fuente["sha256"] is None:
        advertencias.append("Fuente en vivo: no hay hash del video para reproducir el analisis")

    return {
        "frames": frames,
        "curva_desplaz_carga": {"x": columna("v_mm"), "y": columna("carga_n")},
        "curva_esfuerzo_strain": {"x": columna("eps_yy"), "y": columna("esfuerzo_mpa")},
        "ensayo": datos_ensayo,
        "mapa_xx_path": "mapa_xx.jpg",
        "mapa_yy_path": "mapa_yy.jpg",
        "mapa_xy_path": "mapa_xy.jpg",
        "esfuerzo_max_mpa": _max_finito(columna("esfuerzo_mpa")),
        "strain_max": _max_finito([abs(v) for v in columna("eps_yy")]),
        "coef_correlacion_medio": float(np.mean(columna("coef_correlacion"))),
        "carga": carga_info,
        "advertencias": advertencias,
        "metadatos": {
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "software": {"version": VERSION},
            "parametros": {
                "subset_size": cfg.subset_size,
                "search_range": cfg.search_range,
                "paso_nodos_px": _paso_px(geometria.nodos, cfg.columnas),
            },
            "malla": {
                "filas": cfg.filas,
                "columnas": cfg.columnas,
                "subset_malla_px": cfg.subset_malla,
                "nodos_total": len(geometria.nodos),
                "nodos_validos": geometria.validos.count(True),
                "triangulos": len(geometria.triangulos),
                "poligono_px": cfg.poligono,
            },
            "calibracion": {
                "long_calibracion_px": cfg.medidas.long_calibracion_px,
                "long_calibracion_mm": cfg.medidas.long_calibracion_mm,
                "mm_por_px": probeta.mm_por_px,
                "ancho_mm": cfg.medidas.ancho_mm,
                "espesor_mm": cfg.medidas.espesor_mm,
                "area_mm2": probeta.area_mm2,
                "long_calibre_mm": cfg.medidas.long_calibre_mm,
            },
            "video": info_fuente,
            "carga": {
                "metodo": cfg.fuente_carga,
                "roi_carga": cfg.roi_carga if cfg.fuente_carga == "ocr" else None,
                "archivo": None
                if serie is None
                else {
                    "nombre": serie.nombre_archivo,
                    "sha256": serie.sha256,
                    "mapeo": serie.mapeo,
                    "sincronizacion": serie.sincronizacion,
                },
            },
            "ensayo": {"tipo": ensayo.tipo, "parametros": cfg.parametros_ensayo, "puntos_px": cfg.puntos_ensayo},
            "limitacion": "DIC 2D con una camara: el movimiento fuera del plano introduce error",
        },
    }
