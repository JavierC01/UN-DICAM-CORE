"""UN-DICAM CORE - punto de entrada de la herramienta DIC 2D.

Uso:
    python main.py                                              # asistente paso a paso
    python main.py --config resultados/P-01_.../config_usada.json   # repetir un analisis sin preguntas

El asistente pide la identificacion de la probeta, el video, sus medidas, calibra px/mm con dos puntos de
distancia conocida, identifica la probeta en el video para generar la malla y ejecuta el analisis.
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

import math

import numpy as np

from dic.analisis import AnalisisError, ConfigAnalisis, SerieCarga, ejecutar_analisis, sin_nan
from dic.calibracion import calibrar
from dic.fuentes import (
    FuenteError,
    contar_muestras,
    frame_camara,
    frame_video,
    info_video,
    listar_camaras,
    memoria_estimada_mb,
    paso_frames,
    sha256_archivo,
)
from dic.cargas.importador import UNIDADES_CARGA, UNIDADES_TIEMPO, ImportacionCargaError, extraer_serie, leer_tabla
from dic.consola import pedir_float, pedir_int, pedir_opcion, pedir_si_no, pedir_texto
from dic.ensayos import ENSAYOS, obtener_ensayo
from dic.export import exportar_csv, exportar_jpg, exportar_pdf
from dic.interactivo import dibujar_malla, marcar_puntos, marcar_rectangulo
from dic.mallado import identificar_probeta
from dic.probeta import MedidasProbeta
from dic.strain import localizar_punto

CARPETA_VIDEOS = RAIZ / "videos"
CARPETA_CARGAS = RAIZ / "cargas"
EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".3gp")
EXT_CARGA = (".csv", ".txt", ".xlsx")


# ----------------------------------------------------------------------------- busqueda de archivos

def buscar_archivo(texto: str, carpeta: Path, extensiones: tuple[str, ...]) -> Path | None:
    """Acepta ruta completa, ruta relativa o solo el nombre (con o sin extension) dentro de `carpeta`."""
    ruta = Path(texto.strip().strip('"').strip("'"))
    bases = [ruta] if ruta.is_absolute() else [Path.cwd() / ruta, RAIZ / ruta, carpeta / ruta]
    for base in bases:
        candidatos = [base] + ([base.with_name(base.name + e) for e in extensiones] if not base.suffix else [])
        for c in candidatos:
            if c.is_file():
                return c.resolve()
    return None


def elegir_archivo(titulo: str, carpeta: Path, extensiones: tuple[str, ...]) -> Path:
    disponibles = sorted(p for p in carpeta.glob("*") if p.suffix.lower() in extensiones) if carpeta.is_dir() else []
    if disponibles:
        print(f"   Archivos en {carpeta.name}/:")
        for i, p in enumerate(disponibles, 1):
            print(f"     {i}) {p.name}")
    ayuda = "numero de la lista, nombre del archivo o ruta completa" if disponibles else "nombre del archivo o ruta completa"
    while True:
        texto = pedir_texto(f"{titulo} ({ayuda})")
        if disponibles and texto.isdigit() and 1 <= int(texto) <= len(disponibles):
            return disponibles[int(texto) - 1].resolve()
        ruta = buscar_archivo(texto, carpeta, extensiones)
        if ruta is not None:
            return ruta
        print(f"   No se encontro '{texto}'. Revisa el nombre o pega la ruta completa.")


# ----------------------------------------------------------------------------- asistente

def paso(n: int, titulo: str) -> None:
    print(f"\n[{n}] {titulo}\n" + "-" * (len(titulo) + 4))


def asistente() -> dict:
    print("=" * 60 + "\n UN-DICAM CORE - Analisis DIC 2D\n" + "=" * 60)
    print(" Enter acepta el valor entre [corchetes]. En las ventanas: Esc cancela.")

    paso(1, "Identificacion de la probeta")
    id_probeta = pedir_texto("Identificacion de la probeta (ej. P-01)")
    descripcion = input("   Descripcion / material (opcional): ").strip()

    paso(2, "Fuente de imagenes y datos por segundo")
    fuente, frame = elegir_fuente()

    paso(3, "Tipo de ensayo")
    tipo = pedir_opcion("Tipo de ensayo", [(t, e.nombre) for t, e in ENSAYOS.items()])
    ensayo = obtener_ensayo(tipo)
    parametros = {}
    if tipo == "cortante":
        parametros["eje_carga"] = pedir_opcion("Direccion de la carga", [("y", "vertical (y)"), ("x", "horizontal (x)")])
    elif tipo == "portico":
        parametros["brazo_mm"] = pedir_float("Brazo: distancia de la carga al nudo (mm)")

    paso(4, "Medidas de la probeta (mm)")
    probeta = {
        "ancho_mm": pedir_float("Ancho de la seccion (mm)"),
        "espesor_mm": pedir_float("Espesor de la seccion (mm)"),
        "long_calibre_mm": pedir_float("Longitud de calibre (mm)"),
    }
    print(f"   Area de la seccion: {probeta['ancho_mm'] * probeta['espesor_mm']:.3f} mm2")

    paso(5, "Calibracion px -> mm")
    cal = calibrar(frame)

    paso(6, "Identificacion de la probeta y malla de analisis")
    subset = pedir_int("Tamano de subset de correlacion (px, impar recomendado)", 31, 9, 201)
    malla, geometria = identificar_probeta(frame, subset)

    paso(7, f"Puntos del extensometro virtual ({ensayo.nombre})")
    fondo = dibujar_malla(frame, malla["poligono_px"], geometria.nodos, geometria.validos, geometria.triangulos)
    while True:
        print(f"   Marca {ensayo.n_puntos} puntos: {ensayo.puntos}. Deben quedar dentro de la malla verde.")
        puntos = marcar_puntos(fondo, f"Ensayo de {ensayo.nombre}", ensayo.n_puntos)
        fuera = [i + 1 for i, p in enumerate(puntos) if localizar_punto(p, geometria.nodos, geometria.triangulos) is None]
        if not fuera:
            break
        print(f"   Los puntos {fuera} quedan fuera de la malla; marcalos de nuevo.")

    paso(8, "Fuente de carga")
    carga = elegir_carga(frame)

    paso(9, "Parametros del analisis")
    search_range = pedir_int("Rango de busqueda por frame (px, desplazamiento maximo esperado entre frames)", 20, 2, 200)
    ajuste = None
    if carga["fuente"] != "ninguna" and pedir_si_no("Calcular la propiedad del ensayo por ajuste lineal", True):
        print(f"   Rango del ajuste como fraccion del valor maximo de {ensayo.respuesta.nombre.lower()} (zona elastica).")
        fmin = pedir_float("Fraccion minima", 0.1, positivo=False)
        fmax = pedir_float("Fraccion maxima", 0.4)
        ajuste = {"fraccion_min": fmin, "fraccion_max": fmax}

    return {
        "id_probeta": id_probeta,
        "descripcion": descripcion,
        "fuente": fuente,
        "calibracion": {
            "puntos_px": cal.puntos_px,
            "long_calibracion_px": cal.distancia_px,
            "long_calibracion_mm": cal.distancia_mm,
            "px_por_mm": cal.px_por_mm,
        },
        "probeta": probeta,
        "malla": malla,
        "analisis": {"subset_size": subset, "search_range": search_range},
        "ensayo": {"tipo": tipo, "parametros": parametros, "puntos_px": puntos, "ajuste": ajuste},
        "carga": carga,
    }


def elegir_fuente() -> tuple[dict, np.ndarray]:
    """Archivo de video o camara (solo si el equipo tiene una) y cuantos datos por segundo se necesitan."""
    print("   Buscando camaras conectadas...")
    camaras = listar_camaras()
    if camaras:
        etiqueta_cam = f"Camara en vivo ({len(camaras)} detectada{'s' if len(camaras) > 1 else ''})"
        motivo = None
    else:
        etiqueta_cam, motivo = "Camara en vivo", "no se detecto ninguna camara en este equipo"
    tipo = pedir_opcion("Fuente de las imagenes", [("archivo", "Archivo de video"), ("camara", etiqueta_cam, motivo)])
    return elegir_video() if tipo == "archivo" else elegir_camara(camaras)


def elegir_video() -> tuple[dict, np.ndarray]:
    while True:
        ruta = elegir_archivo("Video", CARPETA_VIDEOS, EXT_VIDEO)
        try:
            info = info_video(ruta)
        except FuenteError as exc:
            print(f"   {exc}")
            continue
        if info.fps > 0 and info.n_frames > 1:
            break
        print("   El video no informa fps o numero de frames; no se puede descomponer. Prueba con otro archivo.")
    fps, ultimo = info.fps, info.n_frames - 1
    print(f"   {info.ancho}x{info.alto} px | {fps:g} fps | {info.n_frames} frames | {info.duracion_s:.2f} s")

    while True:
        t_ini = pedir_float("Inicio del analisis en s (imagen de referencia)", 0.0, positivo=False)
        t_fin = pedir_float("Fin del analisis en s", round(ultimo / fps, 3))
        frame_inicio = max(0, min(ultimo - 1, round(t_ini * fps)))
        frame_fin = max(frame_inicio + 1, min(ultimo, round(t_fin * fps)))
        print(f"   Necesitas pocos datos para un ensayo lento y mas datos para uno rapido (maximo {fps:g} = todos los frames).")
        dps = pedir_float("Datos por segundo necesarios", fps)
        if dps > fps:
            print(f"   El video solo tiene {fps:g} frames por segundo; no se pueden obtener {dps:g} datos/s.")
            continue
        salto = paso_frames(fps, dps)
        total = contar_muestras(frame_inicio, frame_fin, salto)
        if total < 1:
            print("   Con ese rango y esos datos por segundo no queda ninguna imagen; amplia el rango o pide mas datos.")
            continue
        real = fps / salto
        if not math.isclose(real, dps, rel_tol=1e-3):
            print(f"   Con {fps:g} fps se toma 1 de cada {salto} frames: {real:.4g} datos/s (lo mas cercano a {dps:g}).")
        else:
            print(f"   Se toma 1 de cada {salto} frames ({real:.4g} datos/s).")
        print(f"   Se analizaran {total + 1} imagenes (frames {frame_inicio} a {frame_fin}, cada {1 / real:.4g} s).")
        break

    fuente = {"tipo": "archivo", "ruta": str(ruta), "frame_inicio": frame_inicio, "frame_fin": frame_fin, "datos_por_segundo": dps}
    return fuente, frame_video(ruta, frame_inicio)


def elegir_camara(camaras) -> tuple[dict, np.ndarray]:
    opciones = [(i, f"Camara {i}: {c.ancho}x{c.alto} px, {c.fps:g} fps") for i, c in camaras]
    if len(camaras) > 1:
        indice = pedir_opcion("Camara a usar", opciones)
    else:
        indice = camaras[0][0]
        print(f"   Se usara la {opciones[0][1]}")
    info = dict(camaras)[indice]
    maximo = info.fps if info.fps > 0 else 60.0
    limite = f"la camara entrega hasta {maximo:g}" if info.fps > 0 else "la camara no informa sus fps; el ritmo real se mide al capturar"
    while True:
        dps = pedir_float(f"Datos por segundo necesarios ({limite})", min(10.0, maximo))
        if dps > maximo:
            print(f"   La camara no entrega mas de {maximo:g} imagenes por segundo.")
            continue
        duracion = pedir_float("Duracion de la captura (s)", 60.0)
        n = int(round(dps * duracion)) + 1
        memoria = memoria_estimada_mb(info, n)
        print(f"   Se capturaran {n} imagenes (cada {1 / dps:.4g} s), ~{memoria:,.0f} MB en memoria.")
        if memoria > 4000 and not pedir_si_no("Es mucha memoria; continuar de todos modos", False):
            continue
        break
    print("   Tomando imagen de la camara para calibrar y marcar (no muevas la camara despues de esto)...")
    fuente = {"tipo": "camara", "indice": indice, "datos_por_segundo": dps, "duracion_s": duracion}
    return fuente, frame_camara(indice)


def elegir_carga(frame) -> dict:
    fuente = pedir_opcion(
        "De donde se obtiene la carga",
        [("ninguna", "sin carga (solo desplazamientos y deformaciones)"),
         ("archivo", "archivo exportado por la maquina (CSV/TXT/XLSX)"),
         ("ocr", "lectura de la pantalla de la maquina en el video (OCR)")],
    )
    if fuente == "ninguna":
        return {"fuente": "ninguna"}
    if fuente == "ocr":
        if shutil.which("tesseract") is None:
            print("   AVISO: Tesseract no esta instalado o no esta en el PATH; el OCR fallara.")
        return {"fuente": "ocr", "roi": marcar_rectangulo(frame, "Pantalla de carga: 2 esquinas opuestas")}

    while True:
        ruta = elegir_archivo("Archivo de carga", CARPETA_CARGAS, EXT_CARGA)
        try:
            tabla = leer_tabla(ruta.read_bytes(), ruta.name)
            break
        except ImportacionCargaError as exc:
            print(f"   {exc}")
    columnas = list(tabla.columns)
    print(f"   {len(tabla)} filas. Primeras filas:\n" + "\n".join("     " + l for l in tabla.head(3).to_string().splitlines()))
    col_t = pedir_opcion("Columna de TIEMPO", [(c, c) for c in columnas], 1)
    col_c = pedir_opcion("Columna de CARGA", [(c, c) for c in columnas], min(2, len(columnas)))
    u_t = pedir_opcion("Unidad del tiempo", [(u, u) for u in UNIDADES_TIEMPO], 1)
    u_c = pedir_opcion("Unidad de la carga", [(u, u) for u in UNIDADES_CARGA], 2)
    evento = pedir_opcion(
        "Sincronizacion: el frame de referencia coincide con",
        [("ninguno", "el inicio del registro de la maquina"), ("umbral_carga", "el instante en que la carga supera un umbral")],
    )
    sinc = {"evento_inicio": evento, "umbral_n": None, "offset_s": pedir_float("Desfase adicional (s)", 0.0, positivo=False)}
    if evento == "umbral_carga":
        sinc["umbral_n"] = pedir_float("Umbral de carga (N)")
    return {
        "fuente": "archivo",
        "archivo": {"ruta": str(ruta), "columna_tiempo": col_t, "columna_carga": col_c, "unidad_tiempo": u_t, "unidad_carga": u_c},
        "sincronizacion": sinc,
    }


# ----------------------------------------------------------------------------- ejecucion

def cargar_serie(carga: dict) -> SerieCarga:
    arch = carga["archivo"]
    ruta = Path(arch["ruta"])
    if not ruta.is_file():
        raise ImportacionCargaError(f"No existe el archivo de carga: {ruta}")
    mapeo = {k: arch[k] for k in ("columna_tiempo", "columna_carga", "unidad_tiempo", "unidad_carga")}
    tiempos, cargas, descartadas = extraer_serie(leer_tabla(ruta.read_bytes(), ruta.name), **mapeo)
    if descartadas:
        print(f"   Archivo de carga: {descartadas} filas no numericas descartadas")
    return SerieCarga(tiempos, cargas, ruta.name, sha256_archivo(ruta), mapeo, carga.get("sincronizacion") or {})


def construir_config(cfg: dict) -> ConfigAnalisis:
    cal, prob, malla, an, ens, carga, fte = (
        cfg[k] for k in ("calibracion", "probeta", "malla", "analisis", "ensayo", "carga", "fuente")
    )
    camara = fte["tipo"] == "camara"
    return ConfigAnalisis(
        fuente=int(fte["indice"]) if camara else fte["ruta"],
        datos_por_segundo=fte.get("datos_por_segundo"),
        frame_inicio=0 if camara else int(fte["frame_inicio"]),
        frame_fin=None if camara else fte.get("frame_fin"),
        duracion_s=float(fte.get("duracion_s", 10.0)),
        medidas=MedidasProbeta(
            long_calibracion_px=float(cal["long_calibracion_px"]),
            long_calibracion_mm=float(cal["long_calibracion_mm"]),
            ancho_mm=float(prob["ancho_mm"]),
            espesor_mm=float(prob["espesor_mm"]),
            long_calibre_mm=float(prob["long_calibre_mm"]),
        ),
        poligono=malla["poligono_px"],
        filas=int(malla["filas"]),
        columnas=int(malla["columnas"]),
        subset_malla=int(malla["subset_size"]),
        subset_size=int(an["subset_size"]),
        search_range=int(an["search_range"]),
        tipo_ensayo=ens["tipo"],
        parametros_ensayo=ens.get("parametros") or {},
        puntos_ensayo=ens.get("puntos_px"),
        ajuste=ens.get("ajuste"),
        fuente_carga=carga["fuente"],
        roi_carga=carga.get("roi"),
        serie_carga=cargar_serie(carga) if carga["fuente"] == "archivo" else None,
    )


def crear_barra(unidad: str = "imagenes"):
    inicio = time.time()

    def barra(idx: int, total: int) -> None:
        llenos = int(30 * idx / total)
        print(f"\r   [{'#' * llenos}{'.' * (30 - llenos)}] {idx}/{total} {unidad} ({time.time() - inicio:.0f} s)", end="", flush=True)
        if idx == total:
            print()

    return barra


def main() -> int:
    parser = argparse.ArgumentParser(description="UN-DICAM CORE - analisis DIC 2D")
    parser.add_argument("--config", help="config_usada.json de un analisis anterior, para repetirlo sin preguntas")
    args = parser.parse_args()

    try:
        if args.config:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        else:
            cfg = asistente()
        config = construir_config(cfg)
        nombre = re.sub(r"[^\w.-]+", "_", cfg["id_probeta"]).strip("_") or "probeta"
        salida = RAIZ / "resultados" / f"{nombre}_{time.strftime('%Y%m%d_%H%M%S')}"
        if cfg["fuente"]["tipo"] == "camara":
            input("\nTodo listo. Presiona Enter para INICIAR LA CAPTURA (Esc en la ventana la termina antes)... ")
            print("Capturando; al terminar empieza el analisis.")
        else:
            print("\nProcesando (la primera ejecucion compila numba, puede tardar unos segundos)...")
        data = ejecutar_analisis(
            config, salida, progreso=crear_barra("imagenes analizadas"), progreso_captura=crear_barra("imagenes capturadas")
        )
    except (AnalisisError, FuenteError, ImportacionCargaError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"\nError: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError) as exc:
        print(f"\nCancelado. {exc}")
        return 1

    data["metadatos"]["probeta"] = {"id": cfg["id_probeta"], "descripcion": cfg.get("descripcion", "")}
    print("\nExportando resultados...")
    (salida / "config_usada.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (salida / "resultados.json").write_text(json.dumps(sin_nan(data), ensure_ascii=False, indent=1), encoding="utf-8")
    (salida / "resultados.csv").write_text(exportar_csv(data), encoding="utf-8")
    (salida / "resumen.jpg").write_bytes(exportar_jpg(data, salida))
    (salida / "reporte.pdf").write_bytes(exportar_pdf(data, salida, cfg["id_probeta"]))

    video = data["metadatos"]["video"]
    print(
        f"\nListo. {len(data['frames'])} imagenes analizadas a {video['datos_por_segundo_reales']:.4g} datos/s, "
        f"correlacion media {data['coef_correlacion_medio']:.3f}"
    )
    ajuste = (data.get("ensayo") or {}).get("ajuste")
    if ajuste:
        prop = data["ensayo"]["propiedad"]
        print(f"   {prop['nombre']}: {ajuste['pendiente']:.4g} {prop['unidad']} (R2 = {ajuste['r2']:.4f})")
    for a in data["advertencias"]:
        print(f"   ADVERTENCIA: {a}")
    print(f"Resultados en: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
