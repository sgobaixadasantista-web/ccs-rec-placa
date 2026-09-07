import cv2
import pytesseract
import re
import sys
from typing import Optional, Tuple

PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
PADRAO_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")


def normalizar_texto(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def placa_valida(placa: str) -> bool:
    return bool(PADRAO_MERCOSUL.fullmatch(placa) or PADRAO_ANTIGO.fullmatch(placa))


def reconhecer_placa(
    caminho_imagem: str,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Imagem não encontrada: {caminho_imagem}")

    if bbox:
        x1, y1, x2, y2 = bbox
        imagem = imagem[y1:y2, x1:x2]

    imagem = cv2.resize(imagem, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    realcada = clahe.apply(cinza)

    binaria = cv2.threshold(
        realcada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    bruto = pytesseract.image_to_string(binaria, config=config).strip()
    placa = normalizar_texto(bruto)

    return {
        "placa": placa,
        "valida": placa_valida(placa),
        "ocr_bruto": bruto,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python reconhecer_placa.py imagem.jpg")
        raise SystemExit(1)

    resultado = reconhecer_placa(sys.argv[1])
    print(resultado)
