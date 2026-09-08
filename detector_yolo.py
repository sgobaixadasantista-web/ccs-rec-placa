import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

MODELO_PADRAO = (
    "https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt"
)

PADRAO_ANTIGO = re.compile(r"^[A-Z]{3}[0-9]{4}$")
PADRAO_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Alternativas plausiveis do OCR com custo relativo de correcao.
# Quanto menor o custo, mais provavel a substituicao.
# Em placas, o EasyOCR confunde com frequencia o algarismo 7 com a letra Z;
# por isso Z->7 recebe custo menor que Z->2 em posicoes numericas.
CONFUSOES_NUM = {
    "O": (("0", 1.0),),
    "Q": (("0", 1.0),),
    "D": (("0", 1.0),),
    "I": (("1", 1.0),),
    "L": (("1", 1.0),),
    "Z": (("7", 0.7), ("2", 1.0)),
    "S": (("5", 1.0),),
    "B": (("8", 1.0),),
    "G": (("6", 1.0),),
}

CONFUSOES_LETRA = {
    "0": (("O", 1.0),),
    "1": (("I", 1.0),),
    "2": (("Z", 1.0),),
    "5": (("S", 1.0),),
    "8": (("B", 1.0),),
    "6": (("G", 1.0),),
}

_READER = None


def obter_reader():
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["en"], gpu=False)
    return _READER


def limpar(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def _opcoes_posicao(c: str, tipo: str):
    if tipo == "L":
        if c.isalpha():
            return [(c, 0.0)]
        return list(CONFUSOES_LETRA.get(c, ()))

    if c.isdigit():
        return [(c, 0.0)]
    return list(CONFUSOES_NUM.get(c, ()))


def candidatos_por_mascara(texto: str, mascara: str, modelo: str):
    texto = limpar(texto)
    if len(texto) != len(mascara):
        return []

    candidatos = [("", 0.0)]
    for c, tipo in zip(texto, mascara):
        opcoes = _opcoes_posicao(c, tipo)
        if not opcoes:
            return []

        novos = []
        for prefixo, custo in candidatos:
            for caractere, incremento in opcoes:
                novos.append((prefixo + caractere, custo + incremento))
        candidatos = novos

    saida = []
    for placa, custo in candidatos:
        valida = (
            modelo == "ANTIGA" and PADRAO_ANTIGO.fullmatch(placa)
        ) or (
            modelo == "MERCOSUL" and PADRAO_MERCOSUL.fullmatch(placa)
        )
        if valida:
            saida.append({
                "placa": placa,
                "modelo": modelo,
                "valida": True,
                "correcoes": round(custo, 3),
            })
    return saida


def _janelas_placa(texto: str):
    """Gera trechos de 7 caracteres para ignorar texto extra capturado pelo OCR."""
    texto = limpar(texto)
    if len(texto) < 7:
        return []
    if len(texto) == 7:
        return [(texto, 0)]
    return [(texto[i:i + 7], i) for i in range(len(texto) - 6)]


def gerar_candidatos_validos(texto: str):
    texto = limpar(texto)
    if len(texto) < 7:
        return []

    # O EasyOCR pode concatenar a placa com adesivos, marca do veiculo ou outros
    # caracteres do recorte. Em vez de rejeitar toda a leitura quando ela tem
    # mais de 7 caracteres, avaliamos cada janela de 7 separadamente.
    melhores = {}
    for janela, offset in _janelas_placa(texto):
        candidatos = []
        candidatos.extend(candidatos_por_mascara(janela, "LLLNNNN", "ANTIGA"))
        candidatos.extend(candidatos_por_mascara(janela, "LLLNLNN", "MERCOSUL"))

        for cand in candidatos:
            # Pequena penalidade apenas para desempatar janelas igualmente boas.
            # O custo principal continua sendo o numero/probabilidade das correcoes.
            cand = dict(cand)
            cand["janela_ocr"] = janela
            cand["offset_ocr"] = offset
            cand["custo_ranking"] = round(cand["correcoes"] + offset * 0.01, 3)
            chave = (cand["placa"], cand["modelo"])
            anterior = melhores.get(chave)
            if anterior is None or cand["custo_ranking"] < anterior["custo_ranking"]:
                melhores[chave] = cand

    saida = list(melhores.values())
    saida.sort(key=lambda x: (x["custo_ranking"], x["correcoes"], x["offset_ocr"]))
    return saida


def classificar_e_corrigir(texto: str):
    texto = limpar(texto)
    candidatos = gerar_candidatos_validos(texto)
    if candidatos:
        melhor = candidatos[0]
        return melhor["placa"], melhor["modelo"], True, melhor["correcoes"]
    return texto, None, False, 99.0


def preprocessamentos(crop):
    maior = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(maior, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(
        clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    adapt = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    return [maior, gray, clahe, otsu, adapt]


def ler_easyocr(img):
    reader = obter_reader()
    resultados = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.25,
    )

    if not resultados:
        return "", 0.0

    resultados = sorted(
        resultados,
        key=lambda r: min(p[0] for p in r[0]),
    )

    partes = []
    confs = []
    for _, texto, confianca in resultados:
        texto = limpar(texto)
        if texto:
            partes.append(texto)
            confs.append(float(confianca))

    bruto = "".join(partes)
    confianca = sum(confs) / len(confs) if confs else 0.0
    return bruto, confianca


def ocr_placa(crop):
    leituras_brutas = []
    agregados = defaultdict(lambda: {
        "suporte": 0,
        "soma_conf": 0.0,
        "melhor_conf": 0.0,
        "menor_custo": 99.0,
        "menor_custo_ranking": 99.0,
        "modelo": None,
        "ocr_brutos": [],
        "janelas_ocr": [],
    })

    for img in preprocessamentos(crop):
        bruto, confianca = ler_easyocr(img)
        bruto = limpar(bruto)
        leituras_brutas.append({
            "texto": bruto,
            "confianca": round(confianca, 4),
        })

        for cand in gerar_candidatos_validos(bruto):
            chave = cand["placa"]
            agg = agregados[chave]
            agg["suporte"] += 1
            agg["soma_conf"] += confianca
            agg["melhor_conf"] = max(agg["melhor_conf"], confianca)
            agg["menor_custo"] = min(agg["menor_custo"], cand["correcoes"])
            agg["menor_custo_ranking"] = min(
                agg["menor_custo_ranking"], cand["custo_ranking"]
            )
            agg["modelo"] = cand["modelo"]
            agg["ocr_brutos"].append(bruto)
            agg["janelas_ocr"].append(cand["janela_ocr"])

    if agregados:
        ranking = []
        for placa, agg in agregados.items():
            media_conf = agg["soma_conf"] / max(agg["suporte"], 1)
            ranking.append({
                "placa": placa,
                "modelo": agg["modelo"],
                "valida": True,
                "confianca_ocr": round(media_conf, 4),
                "correcoes": round(agg["menor_custo"], 3),
                "custo_ranking": round(agg["menor_custo_ranking"], 3),
                "suporte_ocr": agg["suporte"],
                "melhor_confianca_ocr": round(agg["melhor_conf"], 4),
                "janela_ocr": agg["janelas_ocr"][0] if agg["janelas_ocr"] else "",
                "ocr_bruto": max(
                    agg["ocr_brutos"],
                    key=lambda bruto: max(
                        (x["confianca"] for x in leituras_brutas if x["texto"] == bruto),
                        default=0,
                    ),
                ),
                "leituras_ocr": leituras_brutas,
                "motor_ocr": "easyocr",
            })

        ranking.sort(
            key=lambda x: (
                x["suporte_ocr"],
                -x["custo_ranking"],
                x["confianca_ocr"],
                x["melhor_confianca_ocr"],
            ),
            reverse=True,
        )
        return ranking[0]

    # Se nenhuma mascara brasileira for validada, devolve a melhor leitura bruta.
    if leituras_brutas:
        melhor = max(leituras_brutas, key=lambda x: x["confianca"])
        return {
            "placa": melhor["texto"],
            "modelo": None,
            "valida": False,
            "ocr_bruto": melhor["texto"],
            "confianca_ocr": melhor["confianca"],
            "correcoes": 99.0,
            "suporte_ocr": 0,
            "leituras_ocr": leituras_brutas,
            "motor_ocr": "easyocr",
        }

    return {
        "placa": "",
        "modelo": None,
        "valida": False,
        "ocr_bruto": "",
        "confianca_ocr": 0.0,
        "correcoes": 99.0,
        "suporte_ocr": 0,
        "leituras_ocr": [],
        "motor_ocr": "easyocr",
    }


def detectar(caminho_imagem, modelo_yolo=MODELO_PADRAO, conf=0.25, salvar_recortes=False):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Imagem nao encontrada: {caminho_imagem}")

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
        key=lambda x: (
            x["valida"],
            x.get("suporte_ocr", 0),
            -x.get("correcoes", 99.0),
            x["confianca_yolo"],
            x["confianca_ocr"],
        ),
        reverse=True,
    )
    return saida


def main():
    parser = argparse.ArgumentParser(
        description="Deteccao de placas brasileiras com YOLOv11 + EasyOCR"
    )
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
