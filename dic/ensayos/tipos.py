import math

from dic.ensayos.base import Eje, Ensayo, Punto, _longitud, deformados


class Traccion(Ensayo):
    tipo = "traccion"
    nombre = "traccion"
    n_puntos = 2
    puntos = "los dos extremos de la longitud de calibre del extensometro virtual"
    medida = Eje("Deformacion del extensometro", "mm/mm")
    respuesta = Eje("Esfuerzo", "MPa")
    propiedad = Eje("Modulo de elasticidad", "MPa")
    _signo = 1.0

    def medir(self, puntos_mm: list[Punto], desplazamientos_mm: list[Punto], parametros: dict) -> float:
        l0 = _longitud(*puntos_mm)
        if l0 == 0:
            return math.nan
        return self._signo * (_longitud(*deformados(puntos_mm, desplazamientos_mm)) - l0) / l0

    def calcular_respuesta(self, carga_n: float, area_mm2: float, parametros: dict) -> float:
        return carga_n / area_mm2


class Compresion(Traccion):
    """Acortamiento positivo; se espera que la maquina reporte la carga de compresion como positiva."""

    tipo = "compresion"
    nombre = "compresion"
    _signo = -1.0


class Flexion(Ensayo):
    tipo = "flexion"
    nombre = "flexion"
    n_puntos = 3
    puntos = "apoyo izquierdo, centro de la luz y apoyo derecho"
    medida = Eje("Deflexion", "mm")
    respuesta = Eje("Carga", "N")
    propiedad = Eje("Rigidez a flexion", "N/mm")

    def medir(self, puntos_mm: list[Punto], desplazamientos_mm: list[Punto], parametros: dict) -> float:
        # El eje y de la imagen apunta hacia abajo: deflexion positiva = el centro baja respecto a los apoyos.
        (_, va), (_, vc), (_, vb) = desplazamientos_mm
        return vc - (va + vb) / 2


class Cortante(Ensayo):
    tipo = "cortante"
    nombre = "cortante / conexion"
    n_puntos = 2
    puntos = "un punto en cada elemento de la conexion (primero el fijo, luego el que desliza)"
    medida = Eje("Deslizamiento relativo", "mm")
    respuesta = Eje("Carga", "N")
    propiedad = Eje("Modulo de deslizamiento", "N/mm")

    def validar_parametros(self, parametros: dict) -> None:
        if parametros.get("eje_carga", "y") not in ("x", "y"):
            raise ValueError("eje_carga debe ser 'x' o 'y'")

    def medir(self, puntos_mm: list[Punto], desplazamientos_mm: list[Punto], parametros: dict) -> float:
        componente = 0 if parametros.get("eje_carga", "y") == "x" else 1
        return desplazamientos_mm[1][componente] - desplazamientos_mm[0][componente]


class Portico(Ensayo):
    tipo = "portico"
    nombre = "momento en portico"
    n_puntos = 2
    puntos = "dos puntos a lo largo del elemento cuya rotacion se mide"
    medida = Eje("Rotacion", "rad")
    respuesta = Eje("Momento", "N*mm")
    propiedad = Eje("Rigidez rotacional", "N*mm/rad")

    def validar_parametros(self, parametros: dict) -> None:
        brazo = parametros.get("brazo_mm")
        if not isinstance(brazo, (int, float)) or brazo <= 0:
            raise ValueError("El ensayo de portico necesita brazo_mm > 0 (distancia de la carga al nudo)")

    def medir(self, puntos_mm: list[Punto], desplazamientos_mm: list[Punto], parametros: dict) -> float:
        (x1, y1), (x2, y2) = puntos_mm
        (a1, b1), (a2, b2) = deformados(puntos_mm, desplazamientos_mm)
        delta = math.atan2(b2 - b1, a2 - a1) - math.atan2(y2 - y1, x2 - x1)
        return math.atan2(math.sin(delta), math.cos(delta))

    def calcular_respuesta(self, carga_n: float, area_mm2: float, parametros: dict) -> float:
        return carga_n * float(parametros["brazo_mm"])
