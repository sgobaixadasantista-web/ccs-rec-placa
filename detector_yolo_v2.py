import cv2
import numpy as np

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
    return cv2.addWeighted(maior, 1.8, blur, -0.8, 0)


def _ordenar_pontos(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(soma)],
        pts[np.argmin(dif)],
        pts[np.argmax(soma)],
        pts[np.argmax(dif)],
    ], dtype=np.float32)


def _retificar_quadrilatero(crop, pts):
    tl, tr, br, bl = _ordenar_pontos(pts)
    largura = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    altura = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if largura < 60 or altura < 18:
        return None
    if largura < altura:
        largura, altura = altura, largura
    proporcao = largura / max(altura, 1)
    if not 1.8 <= proporcao <= 6.5:
        return None
    destino = np.array([
        [0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]
    ], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), destino)
    return cv2.warpPerspective(crop, matriz, (largura, altura), flags=cv2.INTER_CUBIC)


def _corrigir_perspectiva(crop):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    area_crop = float(h * w)
    if area_crop <= 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    bordas = cv2.Canny(gray, 45, 150)
    bordas = cv2.morphologyEx(
        bordas,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=2,
    )

    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contorno)
        fracao = area / area_crop
        if fracao < 0.18 or fracao > 0.98:
            continue
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.025 * perimetro, True)
        if len(aprox) != 4 or not cv2.isContourConvex(aprox):
            continue
        retificado = _retificar_quadrilatero(crop, aprox.reshape(4, 2))
        if retificado is None:
            continue
        rh, rw = retificado.shape[:2]
        proporcao = rw / max(rh, 1)
        score = fracao - abs(proporcao - 3.2) * 0.03
        candidatos.append((score, retificado))

    if not candidatos:
        return None
    candidatos.sort(key=lambda x: x[0], reverse=True)
    return candidatos[0][1]


def _score_ocr(item):
    valida = 1 if item.get("valida", False) else 0
    correcoes = float(item.get("correcoes", 99.0))
    conf = float(item.get("confianca_ocr", 0.0))
    suporte = min(int(item.get("suporte_ocr", 0)), 5)
    melhor_conf = float(item.get("melhor_confianca_ocr", conf))
    score = (
        valida * 100.0
        - correcoes * 8.0
        + conf * 24.0
        + melhor_conf * 8.0
        + suporte * 2.0
    )
    if str(item.get("variante_crop", "")).startswith("perspectiva"):
        score -= 1.0
    return score


def ocr_placa_v2(crop):
    central = _crop_central(crop)
    perspectiva = _corrigir_perspectiva(crop)
    perspectiva_central = _corrigir_perspectiva(central)

    variantes = [
        ("original", crop),
        ("central", central),
        ("nitido", _crop_nitido(crop)),
        ("central_nitido", _crop_nitido(central)),
    ]
    if perspectiva is not None:
        variantes.extend([
            ("perspectiva", perspectiva),
            ("perspectiva_nitida", _crop_nitido(perspectiva)),
        ])
    if perspectiva_central is not None:
        variantes.extend([
            ("perspectiva_central", perspectiva_central),
            ("perspectiva_central_nitida", _crop_nitido(perspectiva_central)),
        ])

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
