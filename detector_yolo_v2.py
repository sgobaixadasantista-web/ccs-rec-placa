import cv2

import detector_yolo as base


_OCR_BASE = base.ocr_placa


def _crop_central(crop):
    h, w = crop.shape[:2]
    x1 = int(w * 0.04)
    x2 = max(x1 + 1, int(w * 0.96))
    y1 = int(h * 0.12)
    y2 = max(y1 + 1, int(h * 0.92))
    return crop[y1:y2, x1:x2]


def _crop_nitido(crop):
    maior = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(maior, (0, 0), 1.2)
    nitido = cv2.addWeighted(maior, 1.8, blur, -0.8, 0)
    return nitido


def _score_ocr(item):
    valida = 1 if item.get("valida", False) else 0
    correcoes = float(item.get("correcoes", 99.0))
    conf = float(item.get("confianca_ocr", 0.0))
    suporte = min(int(item.get("suporte_ocr", 0)), 5)
    melhor_conf = float(item.get("melhor_confianca_ocr", conf))
    return (
        valida * 100.0
        - correcoes * 8.0
        + conf * 24.0
        + melhor_conf * 8.0
        + suporte * 2.0
    )


def ocr_placa_v2(crop):
    variantes = [
        ("original", crop),
        ("central", _crop_central(crop)),
        ("nitido", _crop_nitido(crop)),
        ("central_nitido", _crop_nitido(_crop_central(crop))),
    ]

    candidatos = []
    for nome, imagem in variantes:
        if imagem is None or imagem.size == 0:
            continue
        item = dict(_OCR_BASE(imagem))
        item["variante_crop"] = nome
        item["score_ocr_v2"] = round(_score_ocr(item), 4)
        candidatos.append(item)

    if not candidatos:
        return _OCR_BASE(crop)

    candidatos.sort(key=_score_ocr, reverse=True)
    melhor = candidatos[0]
    melhor["alternativas_crop"] = [
        {
            "variante": c.get("variante_crop"),
            "placa": c.get("placa"),
            "valida": c.get("valida"),
            "confianca_ocr": c.get("confianca_ocr"),
            "correcoes": c.get("correcoes"),
            "suporte_ocr": c.get("suporte_ocr"),
            "score": c.get("score_ocr_v2"),
        }
        for c in candidatos
    ]
    return melhor


def _score_deteccao(item):
    valida = 1 if item.get("valida", False) else 0
    correcoes = float(item.get("correcoes", 99.0))
    conf_ocr = float(item.get("confianca_ocr", 0.0))
    conf_yolo = float(item.get("confianca_yolo", 0.0))
    suporte = min(int(item.get("suporte_ocr", 0)), 5)

    return (
        valida * 100.0
        - correcoes * 8.0
        + conf_ocr * 30.0
        + conf_yolo * 24.0
        + suporte * 1.5
    )


def detectar(caminho_imagem, modelo_yolo=base.MODELO_PADRAO, conf=0.25, salvar_recortes=False):
    original = base.ocr_placa
    base.ocr_placa = ocr_placa_v2
    try:
        resultados = base.detectar(
            caminho_imagem,
            modelo_yolo=modelo_yolo,
            conf=conf,
            salvar_recortes=salvar_recortes,
        )
    finally:
        base.ocr_placa = original

    for item in resultados:
        item["score_selecao_v2"] = round(_score_deteccao(item), 4)

    resultados.sort(key=_score_deteccao, reverse=True)
    return resultados
