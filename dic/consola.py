"""Preguntas por consola con validacion. Enter acepta el valor por defecto mostrado entre corchetes."""

import os
import sys

if os.name == "nt":
    os.system("")  # activa las secuencias ANSI (tachado, gris) en la consola de Windows

_ANSI = sys.stdout.isatty()


def tachado(texto: str) -> str:
    return f"\x1b[9;90m{texto}\x1b[0m" if _ANSI else texto


def _leer(mensaje: str, defecto) -> str:
    sufijo = f" [{defecto}]" if defecto is not None else ""
    return input(f"   {mensaje}{sufijo}: ").strip()


def pedir_texto(mensaje: str, defecto: str | None = None) -> str:
    while True:
        valor = _leer(mensaje, defecto)
        if valor:
            return valor
        if defecto is not None:
            return defecto
        print("   Este dato es obligatorio.")


def pedir_float(mensaje: str, defecto: float | None = None, positivo: bool = True) -> float:
    while True:
        texto = _leer(mensaje, defecto)
        if not texto and defecto is not None:
            return float(defecto)
        try:
            valor = float(texto.replace(",", "."))
        except ValueError:
            print("   Ingresa un numero (ej. 12.5).")
            continue
        if positivo and valor <= 0:
            print("   El valor debe ser mayor que cero.")
            continue
        return valor


def pedir_int(mensaje: str, defecto: int | None = None, minimo: int = 0, maximo: int | None = None) -> int:
    while True:
        texto = _leer(mensaje, defecto)
        if not texto and defecto is not None:
            return int(defecto)
        if not texto.lstrip("-").isdigit():
            print("   Ingresa un numero entero.")
            continue
        valor = int(texto)
        if valor < minimo or (maximo is not None and valor > maximo):
            print(f"   El valor debe estar entre {minimo} y {maximo if maximo is not None else 'infinito'}.")
            continue
        return valor


def pedir_opcion(mensaje: str, opciones: list[tuple], defecto: int = 1):
    """opciones: (valor, etiqueta) o (valor, etiqueta, motivo_si_no_disponible). Las no disponibles salen tachadas."""
    print(f"   {mensaje}:")
    habilitadas = []
    for i, (valor, etiqueta, *motivo) in enumerate(opciones, 1):
        bloqueada = bool(motivo and motivo[0])
        if bloqueada:
            print(f"     {tachado(f'{i}) {etiqueta}')}  (no disponible: {motivo[0]})")
        else:
            print(f"     {i}) {etiqueta}")
            habilitadas.append(i)
    if not habilitadas:
        raise ValueError(f"No hay opciones disponibles para: {mensaje}")
    if defecto not in habilitadas:
        defecto = habilitadas[0]
    while True:
        eleccion = pedir_int("Opcion", defecto, 1, len(opciones))
        if eleccion in habilitadas:
            return opciones[eleccion - 1][0]
        print(f"   La opcion {eleccion} no esta disponible.")


def pedir_si_no(mensaje: str, defecto: bool = True) -> bool:
    while True:
        texto = _leer(f"{mensaje} (s/n)", "s" if defecto else "n").lower()
        if not texto:
            return defecto
        if texto in ("s", "si", "sí", "y"):
            return True
        if texto in ("n", "no"):
            return False
