from collections import defaultdict

import detector_yolo as base
import detector_yolo_v2 as v2


def _modelo_placa(placa: str):
    placa = (placa or "").upper()
    if base.PADRAO_MERCOSUL.fullmatch(placa):
        return "MERCOSUL"
    if base.PADRAO_ANTIGO.fullmatch(placa):
        return "ANTIGA"
    return None


def _peso_candidato(item, candidato):
    conf_ocr = float(candidato.get("confianca_ocr", item.get("confianca_ocr", 0.0)) or 0.0)
    conf_yolo = float(item.get("confianca_yolo", 0.0) or 0.0)
    correcoes = float(candidato.get("correcoes", item.get("correcoes", 0.0)) or 0.0)
    suporte = min(int(candidato.get("suporte_ocr", item.get("suporte_ocr", 0)) or 0), 5)
    return max(0.05, 1.0 + conf_ocr * 2.4 + conf_yolo * 1.2 + suporte * 0.12 - correcoes * 0.35)


def _candidatos_por_deteccao(item):
    """Retem somente a melhor evidencia de cada placa por deteccao.

    Isso evita que uma unica bbox com muitas variantes de crop domine o voto.
    """
    melhores = {}

    def adicionar(c):
        placa = str(c.get("placa", "") or "").upper()
        modelo = _modelo_placa(placa)
        if not modelo:
            return
        peso = _peso_candidato(item, c)
        atual = melhores.get(placa)
        if atual is None or peso > atual["peso"]:
            melhores[placa] = {
                "placa": placa,
                "modelo": modelo,
                "peso": peso,
                "confianca_ocr": float(c.get("confianca_ocr", item.get("confianca_ocr", 0.0)) or 0.0),
                "correcoes": float(c.get("correcoes", item.get("correcoes", 0.0)) or 0.0),
            }

    adicionar(item)
    for alt in item.get("alternativas_crop", []) or []:
        if isinstance(alt, dict) and not alt.get("contaminado_faixa", False):
            adicionar(alt)

    return list(melhores.values())


def _consenso_grupo(itens):
    if len(itens) < 2:
        return []

    votos = {
        "ANTIGA": [defaultdict(float) for _ in range(7)],
        "MERCOSUL": [defaultdict(float) for _ in range(7)],
    }
    total_pos = {
        "ANTIGA": [0.0] * 7,
        "MERCOSUL": [0.0] * 7,
    }
    fontes_modelo = defaultdict(set)
    evidencias = []

    for idx, item in enumerate(itens):
        for cand in _candidatos_por_deteccao(item):
            placa = cand["placa"]
            modelo = cand["modelo"]
            peso = cand["peso"]
            fontes_modelo[modelo].add(idx)
            evidencias.append({
                "placa": placa,
                "modelo": modelo,
                "peso": round(peso, 4),
                "confianca_ocr": round(cand["confianca_ocr"], 4),
            })
            for pos, ch in enumerate(placa):
                votos[modelo][pos][ch] += peso
                total_pos[modelo][pos] += peso

    saida = []
    for modelo in ("MERCOSUL", "ANTIGA"):
        if len(fontes_modelo[modelo]) < 2:
            continue
        if any(total <= 0 for total in total_pos[modelo]):
            continue

        chars = []
        acordos = []
        margens = []
        for pos in range(7):
            ordenados = sorted(votos[modelo][pos].items(), key=lambda kv: kv[1], reverse=True)
            if not ordenados:
                chars = []
                break
            ch, melhor = ordenados[0]
            segundo = ordenados[1][1] if len(ordenados) > 1 else 0.0
            total = total_pos[modelo][pos]
            chars.append(ch)
            acordos.append(melhor / total)
            margens.append((melhor - segundo) / total)

        if len(chars) != 7:
            continue
        placa = "".join(chars)
        if _modelo_placa(placa) != modelo:
            continue

        acordo_medio = sum(acordos) / 7
        margem_media = sum(margens) / 7
        # Exige consenso minimamente coerente para nao inventar uma placa a partir
        # de leituras completamente divergentes.
        if acordo_medio < 0.52:
            continue

        x1 = min(i["bbox"][0] for i in itens)
        y1 = min(i["bbox"][1] for i in itens)
        x2 = max(i["bbox"][2] for i in itens)
        y2 = max(i["bbox"][3] for i in itens)
        conf_yolo = max(float(i.get("confianca_yolo", 0.0) or 0.0) for i in itens)

        saida.append({
            "bbox": [x1, y1, x2, y2],
            "confianca_yolo": round(conf_yolo, 4),
            "modelo_detector": itens[0].get("modelo_detector"),
            "placa": placa,
            "modelo": modelo,
            "valida": True,
            "confianca_ocr": round(acordo_medio, 4),
            "melhor_confianca_ocr": round(max(acordos), 4),
            "correcoes": 0.0,
            "suporte_ocr": len(fontes_modelo[modelo]),
            "motor_ocr": "easyocr+consenso",
            "ocr_bruto": "CONSENSO",
            "variante_crop": "consenso_deteccoes",
            "tipo_deteccao": "consenso_ocr",
            "fontes_consenso": len(fontes_modelo[modelo]),
            "acordo_posicional": [round(x, 4) for x in acordos],
            "margem_posicional": [round(x, 4) for x in margens],
            "acordo_medio": round(acordo_medio, 4),
            "margem_media": round(margem_media, 4),
            "evidencias_consenso": evidencias,
        })

    return saida


def _criar_consensos(resultados):
    # Usa somente deteccoes originais; fusoes de imagem podem trazer texto da faixa
    # azul e nao devem votar como se fossem uma nova observacao independente.
    originais = [r for r in resultados if r.get("tipo_deteccao") != "fusao_sobrepostas"]
    saida = []
    for grupo in v2._agrupar_sobrepostas(originais):
        itens = [originais[i] for i in grupo]
        saida.extend(_consenso_grupo(itens))
    return saida


def _score_deteccao_v3(item):
    score = v2._score_deteccao(item)
    if item.get("tipo_deteccao") == "consenso_ocr":
        acordo = float(item.get("acordo_medio", 0.0) or 0.0)
        margem = float(item.get("margem_media", 0.0) or 0.0)
        fontes = min(int(item.get("fontes_consenso", 0) or 0), 4)
        score += acordo * 8.0 + margem * 4.0 + fontes * 1.0
    return score


def detectar(caminho_imagem, modelo_yolo=base.MODELO_PADRAO, conf=0.25, salvar_recortes=False):
    resultados = v2.detectar(
        caminho_imagem,
        modelo_yolo=modelo_yolo,
        conf=conf,
        salvar_recortes=salvar_recortes,
    )

    resultados.extend(_criar_consensos(resultados))
    for item in resultados:
        item["score_selecao_v3"] = round(_score_deteccao_v3(item), 4)
        # Mantem compatibilidade com o resumo existente.
        item["score_selecao_v2"] = item["score_selecao_v3"]

    resultados.sort(key=_score_deteccao_v3, reverse=True)
    return resultados
