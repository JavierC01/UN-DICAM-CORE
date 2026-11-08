import re

import cv2
import numpy as np
import pytesseract

_PATRON_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")


def recortar_roi(imagen: np.ndarray, roi: dict) -> np.ndarray:
    x, y, ancho, alto = int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])
    alto_img, ancho_img = imagen.shape[0], imagen.shape[1]
    if x < 0 or y < 0 or x + ancho > ancho_img or y + alto > alto_img:
        raise ValueError("La region de carga (roi_carga) esta fuera de los limites del frame")
    return imagen[y : y + alto, x : x + ancho]


def preprocesar_para_ocr(recorte: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if recorte.ndim == 3 else recorte
    gris = cv2.resize(gris, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def parsear_valor_numerico(texto: str) -> float:
    coincidencia = _PATRON_NUMERO.search(texto)
    if coincidencia is None:
        raise ValueError(f"No se pudo reconocer un valor numerico de carga en el texto '{texto.strip()}'")
    return float(coincidencia.group().replace(",", "."))


def extract_load_value(imagen: np.ndarray, roi: dict) -> float:
    """Extrae el valor de carga (N) desde la region de pantalla configurada en el frame, via OCR."""
    recorte = recortar_roi(imagen, roi)
    binaria = preprocesar_para_ocr(recorte)
    texto = pytesseract.image_to_string(binaria, config="--psm 7 -c tessedit_char_whitelist=0123456789.,-")
    return parsear_valor_numerico(texto)
