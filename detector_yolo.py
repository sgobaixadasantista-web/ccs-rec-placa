import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

MODELO_PADRAO = (
    "https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt"
)

PADRAO_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")
PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

CONFUSOES_NUM = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6",
}

CONFUSOES_LETRA = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
}


def limpar(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def corrigir_por_mascara(texto: str, mascara: str) -> str:
    if len(texto) != len(mascara):
        return texto

    saida = []
    for c, tipo in zip(texto, mascara):
        if tipo == "L":
            saida.append(CONFUSOES_LETRA.get(c, c))
        else:
            saida.append(CONFUSOES_NUM.get(c, c))
    return "".join(saida)


def classificar_e_corrigir(texto: str):
    texto = limpar(texto)
    candidatos = [
        (corrigir_por_mascara(texto, "LLLNNNN"), "ANTIGA"),
        (corrigir_por_mascara(texto, "LLLNLNN"), "MERCOSUL"),
    ]

    for placa, modelo in candidatos:
        if modelo == "ANTIGA" and PADRAO_ANTIGO.fullmatch(placa):
            return placa, modelo, True
        if modelo == "MERCOSUL" and PADRAO_MERCOSUL.fullmatch(placa):
            return placa, modelo, True

    return texto, None, False


def preprocessamentos(crop):
    maior = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(maior, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adapt = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    return [gray, clahe, otsu, adapt]


def ocr_placa(crop):
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    leituras = []

    for img in preprocessamentos(crop):
        dados = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        partes = []
        confs = []
        for txt, conf in zip(dados["text"], dados["conf"]):
            t = limpar(txt)
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1
            if t:
                partes.append(t)
                if c >= 0:
                    confs.append(c)

        bruto = "".join(partes)
        placa, modelo, valida = classificar_e_corrigir(bruto)
        confianca = float(np.mean(confs) / 100.0) if confs else 0.0
        leituras.append({
            "placa": placa,
            "modelo": modelo,
            "valida": valida,
            "ocr_bruto": bruto,
            "confianca_ocr": round(confianca, 4),
        })

    leituras.sort(
        key=lambda x: (x["valida"], x["confianca_ocr"]),
        reverse=True,
    )
    return leituras[0]


def detectar(caminho_imagem, modelo_yolo=MODELO_PADRAO, conf=0.25, salvar_recortes=False):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Imagem não encontrada: {caminho_imagem}")

    modelo = YOLO(modelo_yolo)
    resultados = modelo.predict(source=imagem, conf=conf, verbose=False)

    saida = []
    pasta = Path("recortes")
    if salvar_recortes:
        pasta.mkdir(exist_ok=True)

    for resultado in resultados:
        if resultado.boxes is None:
            continue

        for i, box in enumerate(resultado.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_yolo = float(box.conf[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(imagem.shape[1], x2)
            y2 = min(imagem.shape[0], y2)

            crop = imagem[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ocr = ocr_placa(crop)
            registro = {
                "bbox": [x1, y1, x2, y2],
                "confianca_yolo": round(conf_yolo, 4),
                "modelo_detector": modelo_yolo,
                **ocr,
            }

            if salvar_recortes:
                destino = pasta / f"placa_{i}_{ocr['placa'] or 'sem_leitura'}.jpg"
                cv2.imwrite(str(destino), crop)
                registro["recorte"] = str(destino)

            saida.append(registro)

    saida.sort(
        key=lambda x: (x["valida"], x["confianca_yolo"], x["confianca_ocr"]),
        reverse=True,
    )
    return saida


def main():
    parser = argparse.ArgumentParser(description="Detecção de placas brasileiras com YOLOv11 + OCR")
    parser.add_argument("imagem", help="Caminho da imagem")
    parser.add_argument(
        "--modelo",
        default=MODELO_PADRAO,
        help="URL ou caminho local para os pesos YOLO",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--salvar-recortes", action="store_true")
    args = parser.parse_args()

    resultados = detectar(
        args.imagem,
        args.modelo,
        conf=args.conf,
        salvar_recortes=args.salvar_recortes,
    )
    print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
