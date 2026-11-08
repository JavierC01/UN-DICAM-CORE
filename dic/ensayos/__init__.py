from dic.ensayos.base import Ensayo, ajuste_lineal
from dic.ensayos.tipos import Compresion, Cortante, Flexion, Portico, Traccion

ENSAYOS: dict[str, Ensayo] = {e.tipo: e for e in (Traccion(), Compresion(), Flexion(), Cortante(), Portico())}
TIPOS_ENSAYO = tuple(ENSAYOS)


def obtener_ensayo(tipo: str) -> Ensayo:
    try:
        return ENSAYOS[tipo]
    except KeyError as exc:
        raise ValueError(f"Tipo de ensayo desconocido: {tipo}. Opciones: {', '.join(TIPOS_ENSAYO)}") from exc


__all__ = ["ENSAYOS", "TIPOS_ENSAYO", "Ensayo", "ajuste_lineal", "obtener_ensayo"]
