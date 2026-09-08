import cv2
import numpy as np
from pathlib import Path

import detector_yolo as base


_OCR_BASE = base.ocr_placa


def _crop_central(crop):
    h, w = crop.shape[:2]
    x1 = int(w * 0.04)
    x2 = max(x1 + 1, int(w * 0.96))
    y1 = int(h * 0.12)
    y2 = max(y1 + 1, int(h * 0.92))
    return crop[y1:y2, x1:x2]


def _crop_sem_faixa_superior(crop):
    """Recorta a faixa superior da placa Mercosul para evitar BRASIL no OCR.

    A variante e apenas adicional: as leituras do crop original continuam sendo
    avaliadas, entao placas antigas ou deteccoes muito apertadas nao dependem
    deste recorte.
    """
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    x1 = int(w * 0.03)
    x2 = max(x1 + 1, int(w * 0.97))
    y1 = int(h * 0.24)
    y2 = max(y1 + 1, int(h * 0.97))
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


def _tem_texto_faixa(item):
    bruto = str(item.get("ocr_bruto", "") or "").upper()
    leituras = item.get("leituras_ocr", []) or []
    textos = [bruto]
    for leitura in leituras:
        if isinstance(leitura, dict):
            textos.append(str(leitura.get("texto", "") or "").upper())
    return any("BRASIL" in texto for texto in textos)


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

    variante = str(item.get("variante_crop", ""))
    if variante.startswith("perspectiva"):
        score -= 1.0
    if variante.startswith("sem_faixa"):
        score += 2.5

    # A palavra BRASIL da faixa azul nao pode virar candidata de placa.
    # Em vez de alterar o OCR-base de forma agressiva, derrubamos fortemente
    # o ranking desta leitura e deixamos variantes sem a faixa competirem.
    if _tem_texto_faixa(item):
        score -= 55.0

    return score


def ocr_placa_v2(crop):
    central = _crop_central(crop)
    sem_faixa = _crop_sem_faixa_superior(crop)
    perspectiva = _corrigir_perspectiva(crop)
    perspectiva_central = _corrigir_perspectiva(central)
    perspectiva_sem_faixa = _corrigir_perspectiva(sem_faixa)

    variantes = [
        ("original", crop),
        ("central", central),
        ("nitido", _crop_nitido(crop)),
        ("central_nitido", _crop_nitido(central)),
        ("sem_faixa", sem_faixa),
        ("sem_faixa_nitido", _crop_nitido(sem_faixa)),
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
    if perspectiva_sem_faixa is not None:
        variantes.extend([
            ("sem_faixa_perspectiva", perspectiva_sem_faixa),
            ("sem_faixa_perspectiva_nitida", _crop_nitido(perspectiva_sem_faixa)),
        ])

    candidatos = []
    for nome, imagem in variantes:
        if imagem is None or imagem.size == 0:
            continue
        item = dict(_OCR_BASE(imagem))
        item["variante_crop"] = nome
        item["contaminado_faixa"] = _tem_texto_faixa(item)
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
            "contaminado_faixa": c.get("contaminado_faixa", False),
            "score": c.get("score_ocr_v2"),
        }
        for c in candidatos
    ]
    return melhor


def _intersecao_sobre_menor(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def _agrupar_sobrepostas(resultados):
    grupos = []
    usados = set()
    for i, item in enumerate(resultados):
        if i in usados:
            continue
        grupo = [i]
        usados.add(i)
        mudou = True
        while mudou:
            mudou = False
            for j, outro in enumerate(resultados):
                if j in usados:
                    continue
                if any(
                    _intersecao_sobre_menor(resultados[k]["bbox"], outro["bbox"]) >= 0.55
                    for k in grupo
                ):
                    grupo.append(j)
                    usados.add(j)
                    mudou = True
        if len(grupo) >= 2:
            grupos.append(grupo)
    return grupos


def _criar_fusoes(caminho_imagem, resultados, salvar_recortes=False):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        return []
    h, w = imagem.shape[:2]
    saida = []

    for numero, grupo in enumerate(_agrupar_sobrepostas(resultados)):
        itens = [resultados[i] for i in grupo]
        x1 = min(i["bbox"][0] for i in itens)
        y1 = min(i["bbox"][1] for i in itens)
        x2 = max(i["bbox"][2] for i in itens)
        y2 = max(i["bbox"][3] for i in itens)

        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * 0.04), int(bh * 0.06)
        fx1, fy1 = max(0, x1 - px), max(0, y1 - py)
        fx2, fy2 = min(w, x2 + px), min(h, y2 + py)
        crop = imagem[fy1:fy2, fx1:fx2]
        if crop.size == 0:
            continue

        ocr = ocr_placa_v2(crop)
        confs = [float(i.get("confianca_yolo", 0.0)) for i in itens]
        registro = {
            "bbox": [fx1, fy1, fx2, fy2],
            "confianca_yolo": round(max(confs) if confs else 0.0, 4),
            "modelo_detector": itens[0].get("modelo_detector"),
            "tipo_deteccao": "fusao_sobrepostas",
            "deteccoes_fundidas": len(itens),
            "bboxes_origem": [i["bbox"] for i in itens],
            **ocr,
        }

        if salvar_recortes:
            pasta = Path("recortes")
            pasta.mkdir(exist_ok=True)
            destino = pasta / f"fusao_{numero}_{ocr.get('placa') or 'sem_leitura'}.jpg"
            cv2.imwrite(str(destino), crop)
            registro["recorte"] = str(destino)

        saida.append(registro)
    return saida


def _bonus_geometria(item):
    x1, y1, x2, y2 = item.get("bbox", [0, 0, 1, 1])
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    proporcao = bw / bh
    distancia = abs(proporcao - 2.8)
    return max(-3.0, 3.0 - distancia * 1.5)


def _score_deteccao(item):
    valida = 1 if item.get("valida", False) else 0
    correcoes = float(item.get("correcoes", 99.0))
    conf_ocr = float(item.get("confianca_ocr", 0.0))
    conf_yolo = float(item.get("confianca_yolo", 0.0))
    suporte = min(int(item.get("suporte_ocr", 0)), 5)
    fusao = 1 if item.get("tipo_deteccao") == "fusao_sobrepostas" else 0
    contaminado = 1 if _tem_texto_faixa(item) else 0

    return (
        valida * 100.0
        - correcoes * 8.0
        + conf_ocr * 30.0
        + conf_yolo * 24.0
        + suporte * 1.5
        + _bonus_geometria(item)
        - fusao * 2.0
        - contaminado * 40.0
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

    resultados.extend(_criar_fusoes(caminho_imagem, resultados, salvar_recortes))

    for item in resultados:
        item["score_selecao_v2"] = round(_score_deteccao(item), 4)

    resultados.sort(key=_score_deteccao, reverse=True)
    return resultados
