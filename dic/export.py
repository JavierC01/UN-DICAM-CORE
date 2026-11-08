"""Exportacion de resultados. Todas las salidas incluyen los metadatos de reproducibilidad del analisis."""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer

COLUMNAS = [
    ("frame", "frame"),
    ("tiempo_s", "tiempo_s"),
    ("u_mm", "u_centro_mm"),
    ("v_mm", "v_centro_mm"),
    ("eps_xx", "eps_xx_medio"),
    ("eps_yy", "eps_yy_medio"),
    ("eps_xy", "eps_xy_medio"),
    ("carga_n", "carga_N"),
    ("esfuerzo_mpa", "esfuerzo_axial_MPa"),
    ("medida_ensayo", "medida_ensayo"),
    ("respuesta_ensayo", "respuesta_ensayo"),
    ("coef_correlacion", "coef_correlacion"),
    ("nodos_perdidos", "nodos_perdidos"),
]


def exportar_csv(data: dict) -> str:
    """Tabla por frame; los metadatos van como comentarios '#' al inicio (pandas.read_csv(comment='#'))."""
    ensayo = data.get("ensayo")
    encabezado = [f"# {linea}" for linea in json.dumps(data.get("metadatos", {}), ensure_ascii=False, indent=1).splitlines()]
    if ensayo:
        encabezado.append(
            f"# medida_ensayo = {ensayo['medida']['nombre']} [{ensayo['medida']['unidad']}]; "
            f"respuesta_ensayo = {ensayo['respuesta']['nombre']} [{ensayo['respuesta']['unidad']}]"
        )
    encabezado += [f"# ADVERTENCIA: {a}" for a in data.get("advertencias", [])]
    tabla = pd.DataFrame(data.get("frames", []))
    tabla = tabla.reindex(columns=[c for c, _ in COLUMNAS]).rename(columns=dict(COLUMNAS))
    return "\n".join(encabezado) + "\n" + tabla.to_csv(index=False)


def _curva(ax, curva: dict, titulo: str, xlabel: str, ylabel: str, ajuste: dict | None = None) -> None:
    puntos = [(x, y) for x, y in zip(curva.get("x", []), curva.get("y", [])) if x is not None and y is not None]
    if puntos:
        xs, ys = zip(*puntos)
        ax.plot(xs, ys, "o-", markersize=3)
        if ajuste:
            ax.plot(xs, [ajuste["pendiente"] * x + ajuste["intercepto"] for x in xs], "r--", linewidth=1, label="ajuste")
            ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Sin datos validos", ha="center", va="center", transform=ax.transAxes, color="red")
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def figura_curvas(data: dict) -> bytes:
    ensayo = data.get("ensayo")
    n = 3 if ensayo else 2
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    _curva(axes[0], data["curva_desplaz_carga"], "Carga - desplazamiento", "Desplazamiento nodo central [mm]", "Carga [N]")
    _curva(axes[1], data["curva_esfuerzo_strain"], "Esfuerzo - deformacion", "eps_yy medio [-]", "Esfuerzo axial [MPa]")
    if ensayo:
        _curva(
            axes[2],
            ensayo["curva"],
            f"Ensayo de {ensayo['nombre']}",
            f"{ensayo['medida']['nombre']} [{ensayo['medida']['unidad']}]",
            f"{ensayo['respuesta']['nombre']} [{ensayo['respuesta']['unidad']}]",
            ensayo.get("ajuste"),
        )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="jpg", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


def _rutas_mapas(data: dict, storage_dir: Path) -> list[Path]:
    rutas = []
    for clave in ("mapa_xx_path", "mapa_yy_path", "mapa_xy_path"):
        relativa = data.get(clave, "").removeprefix("/storage/")
        ruta = storage_dir / relativa
        if relativa and ruta.is_file():
            rutas.append(ruta)
    return rutas


def exportar_jpg(data: dict, storage_dir: Path) -> bytes:
    """Una sola imagen con las curvas arriba y los mapas de deformacion abajo."""
    curvas = plt.imread(io.BytesIO(figura_curvas(data)), format="jpg")
    mapas = [plt.imread(r) for r in _rutas_mapas(data, storage_dir)]
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(curvas)
    ax.axis("off")
    for i, mapa in enumerate(mapas):
        ax = fig.add_subplot(2, max(3, len(mapas)), max(3, len(mapas)) + i + 1)
        ax.imshow(mapa)
        ax.axis("off")
    meta = data.get("metadatos", {})
    fig.suptitle(
        f"UN-DICAM {meta.get('software', {}).get('version', '')} · {meta.get('fecha_utc', '')} · "
        f"video sha256 {str(meta.get('video', {}).get('sha256'))[:12]}",
        fontsize=9,
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="jpg", dpi=110)
    plt.close(fig)
    return buffer.getvalue()


def _fmt(valor, decimales: int = 4) -> str:
    return "sin dato" if valor is None else f"{valor:.{decimales}g}"


def exportar_pdf(data: dict, storage_dir: Path, nombre_proyecto: str) -> bytes:
    estilos = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=f"UN-DICAM {nombre_proyecto}")
    ancho = A4[0] - 3.6 * cm
    historia = [Paragraph(f"Reporte DIC 2D - {nombre_proyecto}", estilos["Title"])]

    ensayo = data.get("ensayo")
    resumen = [
        f"Esfuerzo axial maximo: {_fmt(data.get('esfuerzo_max_mpa'))} MPa",
        f"Deformacion eps_yy maxima: {_fmt(data.get('strain_max'))}",
        f"Coeficiente de correlacion medio: {_fmt(data.get('coef_correlacion_medio'), 3)}",
        f"Fuente de carga: {data.get('carga', {}).get('fuente', 'sin dato')} "
        f"({data.get('carga', {}).get('frames_con_carga', 0)} frames con carga)",
    ]
    if ensayo:
        resumen.append(f"Ensayo: {ensayo['nombre']}")
        ajuste = ensayo.get("ajuste")
        if ajuste:
            resumen.append(
                f"{ensayo['propiedad']['nombre']}: {_fmt(ajuste['pendiente'])} {ensayo['propiedad']['unidad']} "
                f"(R² = {_fmt(ajuste['r2'], 4)}, {ajuste['n_puntos']} puntos, rango {ajuste['rango']})"
            )
        else:
            resumen.append(f"{ensayo['propiedad']['nombre']}: no calculado (define ensayo.ajuste en config.json)")
    historia += [Paragraph("Resumen", estilos["Heading2"])] + [Paragraph(t, estilos["Normal"]) for t in resumen]

    if data.get("advertencias"):
        historia.append(Paragraph("Advertencias", estilos["Heading2"]))
        historia += [Paragraph(f"• {a}", estilos["Normal"]) for a in data["advertencias"]]

    curvas = figura_curvas(data)
    historia += [Paragraph("Curvas", estilos["Heading2"]), Image(io.BytesIO(curvas), width=ancho, height=ancho * 3.6 / (4.2 * (3 if ensayo else 2)))]

    mapas = _rutas_mapas(data, storage_dir)
    if mapas:
        historia.append(Paragraph("Mapas de deformacion (ultimo frame)", estilos["Heading2"]))
        for ruta in mapas:
            historia.append(Image(str(ruta), width=ancho * 0.6, height=ancho * 0.6, kind="proportional"))

    historia += [
        Spacer(1, 0.4 * cm),
        Paragraph("Metadatos de reproducibilidad", estilos["Heading2"]),
        Preformatted(json.dumps(data.get("metadatos", {}), ensure_ascii=False, indent=1), estilos["Code"]),
    ]
    doc.build(historia)
    return buffer.getvalue()
