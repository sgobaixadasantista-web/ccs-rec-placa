import argparse
import shutil
from pathlib import Path

from datasets import load_dataset

DATASET_PADRAO = "justjuu/license-plate-detection"
SPLITS = {"train": "train", "validation": "val", "valid": "val", "test": "test"}


def limpar_destino(destino: Path):
    if destino.exists():
        shutil.rmtree(destino)
    for split in ("train", "val", "test"):
        (destino / "images" / split).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / split).mkdir(parents=True, exist_ok=True)


def extrair_bboxes(exemplo):
    objetos = exemplo.get("objects") or {}
    return objetos.get("bbox") or []


def coco_para_yolo(bbox, largura, altura):
    x, y, w, h = [float(v) for v in bbox[:4]]
    x = max(0.0, min(x, largura))
    y = max(0.0, min(y, altura))
    w = max(0.0, min(w, largura - x))
    h = max(0.0, min(h, altura - y))
    if w <= 0 or h <= 0:
        return None
    return (
        (x + w / 2.0) / largura,
        (y + h / 2.0) / altura,
        w / largura,
        h / altura,
    )


def preparar(dataset_id: str, destino: Path, max_por_split: int = 0):
    print(f"Carregando dataset publico do Hugging Face: {dataset_id}")
    ds = load_dataset(dataset_id)
    print(f"Splits encontrados: {list(ds.keys())}")

    limpar_destino(destino)
    totais = {"train": 0, "val": 0, "test": 0}
    ignoradas = 0

    for split_origem, conjunto in ds.items():
        split_destino = SPLITS.get(split_origem.lower())
        if not split_destino:
            print(f"Ignorando split desconhecido: {split_origem}")
            continue

        limite = len(conjunto) if max_por_split <= 0 else min(len(conjunto), max_por_split)
        print(f"Convertendo {split_origem} -> {split_destino}: {limite} amostras")

        for i in range(limite):
            exemplo = conjunto[i]
            imagem = exemplo.get("image")
            if imagem is None:
                ignoradas += 1
                continue

            imagem = imagem.convert("RGB")
            largura, altura = imagem.size
            if largura <= 0 or altura <= 0:
                ignoradas += 1
                continue

            linhas = []
            for bbox in extrair_bboxes(exemplo):
                if len(bbox) < 4:
                    continue
                yolo = coco_para_yolo(bbox, largura, altura)
                if yolo is None:
                    continue
                xc, yc, w, h = yolo
                linhas.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not linhas:
                ignoradas += 1
                continue

            nome = f"{split_destino}_{i:06d}.jpg"
            img_path = destino / "images" / split_destino / nome
            lbl_path = destino / "labels" / split_destino / f"{Path(nome).stem}.txt"
            imagem.save(img_path, format="JPEG", quality=95)
            lbl_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
            totais[split_destino] += 1

    if totais["train"] == 0 or totais["val"] == 0:
        raise RuntimeError(f"Dataset invalido para treino: {totais}")

    Path("dataset.yaml").write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )

    print("Dataset preparado com sucesso.")
    print(f"train: {totais['train']} imagens")
    print(f"val:   {totais['val']} imagens")
    print(f"test:  {totais['test']} imagens")
    print(f"ignoradas/sem bbox: {ignoradas}")
    print("Classe final: 0 = license_plate")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa dataset publico de placas do Hugging Face e converte COCO para YOLO."
    )
    parser.add_argument("--dataset", default=DATASET_PADRAO)
    parser.add_argument("--destino", default="dataset")
    parser.add_argument(
        "--max-por-split",
        type=int,
        default=0,
        help="0 usa todas as amostras; valor positivo limita cada split para teste rapido.",
    )
    args = parser.parse_args()
    preparar(args.dataset, Path(args.destino), args.max_por_split)


if __name__ == "__main__":
    main()
