import argparse
import re
import shutil
from pathlib import Path

import cv2

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def localizar_imagem(txt_path: Path) -> Path | None:
    """Procura a imagem com o mesmo nome-base do arquivo de anotacao."""
    for ext in EXTENSOES_IMAGEM:
        candidato = txt_path.with_suffix(ext)
        if candidato.exists():
            return candidato
        candidato_upper = txt_path.with_suffix(ext.upper())
        if candidato_upper.exists():
            return candidato_upper
    return None


def extrair_bbox_placa(texto: str):
    """
    Extrai a linha de posicao da placa do formato textual do UFPR-ALPR.

    O dataset possui um arquivo TXT por imagem com informacoes do veiculo,
    identificacao da placa, posicao da placa e posicoes dos caracteres.
    Esta funcao procura variantes comuns como:
      - position_plate
      - position plate
      - plate position
      - license plate position

    Retorna os quatro primeiros numeros encontrados na linha da posicao.
    """
    padroes = (
        r"position[_\s-]*plate",
        r"plate[_\s-]*position",
        r"license[_\s-]*plate[_\s-]*position",
        r"position[_\s-]*license[_\s-]*plate",
    )

    for linha in texto.splitlines():
        linha_lower = linha.lower()
        if any(re.search(p, linha_lower) for p in padroes):
            numeros = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", linha)]
            if len(numeros) >= 4:
                return numeros[:4]
    return None


def converter_bbox(valores, largura_img, altura_img, formato="xywh"):
    """Converte bbox para YOLO: x_center, y_center, width, height normalizados."""
    a, b, c, d = valores

    if formato == "xywh":
        x, y, w, h = a, b, c, d
    elif formato == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
    else:
        raise ValueError("Formato de bbox deve ser 'xywh' ou 'xyxy'.")

    if w <= 0 or h <= 0:
        raise ValueError(f"BBox invalida: {valores}")

    x_center = (x + w / 2.0) / largura_img
    y_center = (y + h / 2.0) / altura_img
    w_norm = w / largura_img
    h_norm = h / altura_img

    valores_norm = (x_center, y_center, w_norm, h_norm)
    if not all(0.0 <= v <= 1.0 for v in valores_norm):
        raise ValueError(
            f"BBox fora dos limites da imagem: {valores} em {largura_img}x{altura_img}"
        )

    return valores_norm


def converter_arquivo(txt_path: Path, destino_images: Path, destino_labels: Path, bbox_format: str):
    imagem_path = localizar_imagem(txt_path)
    if imagem_path is None:
        return False, "imagem correspondente nao encontrada"

    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        return False, "imagem nao pode ser aberta"

    altura, largura = imagem.shape[:2]
    texto = txt_path.read_text(encoding="utf-8", errors="ignore")
    bbox = extrair_bbox_placa(texto)
    if bbox is None:
        return False, "posicao da placa nao encontrada na anotacao"

    try:
        x, y, w, h = converter_bbox(bbox, largura, altura, bbox_format)
    except ValueError as exc:
        return False, str(exc)

    destino_images.mkdir(parents=True, exist_ok=True)
    destino_labels.mkdir(parents=True, exist_ok=True)

    imagem_destino = destino_images / imagem_path.name
    label_destino = destino_labels / f"{imagem_path.stem}.txt"

    shutil.copy2(imagem_path, imagem_destino)
    label_destino.write_text(
        f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n",
        encoding="utf-8",
    )
    return True, "ok"


def descobrir_splits(raiz: Path):
    """
    Tenta aproveitar a divisao oficial train/test/validation quando presente.
    Caso nao exista, trata a raiz inteira como 'train'.
    """
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "validation", "valid"),
        "test": ("test", "testing"),
    }

    encontrados = {}
    for destino, nomes in aliases.items():
        for nome in nomes:
            pasta = raiz / nome
            if pasta.is_dir():
                encontrados[destino] = pasta
                break

    if encontrados:
        return encontrados
    return {"train": raiz}


def main():
    parser = argparse.ArgumentParser(
        description="Converte anotacoes UFPR-ALPR para o formato YOLO (classe license_plate)."
    )
    parser.add_argument(
        "--origem",
        required=True,
        help="Pasta onde o UFPR-ALPR foi extraido localmente.",
    )
    parser.add_argument(
        "--destino",
        default="dataset/ufpr_yolo",
        help="Pasta de saida do dataset convertido.",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("xywh", "xyxy"),
        default="xywh",
        help="Formato dos 4 valores da posicao da placa no TXT. Padrao: xywh.",
    )
    args = parser.parse_args()

    origem = Path(args.origem).expanduser().resolve()
    destino = Path(args.destino).expanduser().resolve()

    if not origem.is_dir():
        raise SystemExit(f"Pasta de origem nao encontrada: {origem}")

    splits = descobrir_splits(origem)
    total_ok = 0
    total_erro = 0

    for split_nome, split_path in splits.items():
        txts = sorted(split_path.rglob("*.txt"))
        print(f"[{split_nome}] {len(txts)} anotacoes encontradas em {split_path}")

        for txt_path in txts:
            ok, motivo = converter_arquivo(
                txt_path,
                destino / "images" / split_nome,
                destino / "labels" / split_nome,
                args.bbox_format,
            )
            if ok:
                total_ok += 1
            else:
                total_erro += 1
                print(f"  IGNORADO: {txt_path} -> {motivo}")

    yaml_path = destino / "dataset_ufpr.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )

    print("\nConversao concluida")
    print(f"- convertidos: {total_ok}")
    print(f"- ignorados/erros: {total_erro}")
    print(f"- destino: {destino}")
    print(f"- YAML: {yaml_path}")


if __name__ == "__main__":
    main()
