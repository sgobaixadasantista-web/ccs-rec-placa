import argparse
import json
from pathlib import Path

import cv2

from detector_yolo import detectar


def desenhar_resultado(imagem_path: str, resultados: list[dict], destino: Path) -> None:
    imagem = cv2.imread(imagem_path)
    if imagem is None:
        raise ValueError(f"Imagem nao encontrada: {imagem_path}")

    for item in resultados:
        x1, y1, x2, y2 = item["bbox"]
        placa = item.get("placa") or "SEM_LEITURA"
        conf_yolo = item.get("confianca_yolo", 0)
        conf_ocr = item.get("confianca_ocr", 0)
        valida = item.get("valida", False)

        cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 255, 255), 2)
        texto = f"{placa} YOLO:{conf_yolo:.2f} OCR:{conf_ocr:.2f}"
        if valida:
            texto += " OK"

        y_texto = max(25, y1 - 8)
        cv2.putText(
            imagem,
            texto,
            (x1, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), imagem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testa o pipeline completo: YOLO treinado + EasyOCR + validacao brasileira"
    )
    parser.add_argument("imagem", help="Caminho da imagem de entrada")
    parser.add_argument("--modelo", required=True, help="Caminho do best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianca minima do YOLO")
    parser.add_argument("--saida", default="resultado_teste", help="Diretorio de saida")
    args = parser.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    resultados = detectar(
        args.imagem,
        args.modelo,
        conf=args.conf,
        salvar_recortes=True,
    )

    json_path = saida / "resultado.json"
    json_path.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    imagem_saida = saida / "resultado_anotado.jpg"
    desenhar_resultado(args.imagem, resultados, imagem_saida)

    resumo_path = saida / "resumo.txt"
    if resultados:
        melhor = resultados[0]
        resumo = (
            f"placa={melhor.get('placa', '')}\n"
            f"valida={melhor.get('valida', False)}\n"
            f"modelo_placa={melhor.get('modelo')}\n"
            f"confianca_yolo={melhor.get('confianca_yolo', 0)}\n"
            f"confianca_ocr={melhor.get('confianca_ocr', 0)}\n"
            f"ocr_bruto={melhor.get('ocr_bruto', '')}\n"
            f"deteccoes={len(resultados)}\n"
        )
    else:
        resumo = "placa=\nvalida=False\ndeteccoes=0\n"

    resumo_path.write_text(resumo, encoding="utf-8")
    print(resumo)
    print(f"JSON: {json_path}")
    print(f"Imagem anotada: {imagem_saida}")


if __name__ == "__main__":
    main()
