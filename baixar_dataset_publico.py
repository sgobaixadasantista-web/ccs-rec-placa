import argparse
import random
import shutil
from pathlib import Path

import kagglehub

DEFAULT_DATASET = "barkataliarbab/license-plate-detection-dataset-10125-images"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}


def baixar(dataset: str) -> Path:
    print(f"Baixando dataset publico via KaggleHub: {dataset}")
    caminho = Path(kagglehub.dataset_download(dataset))
    print(f"Dataset disponivel em: {caminho}")
    return caminho


def normalizar_label(origem: Path, destino: Path) -> bool:
    linhas = []
    for linha in origem.read_text(encoding="utf-8", errors="ignore").splitlines():
        partes = linha.strip().split()
        if len(partes) < 5:
            continue
        try:
            coords = [float(v) for v in partes[1:5]]
        except ValueError:
            continue
        if not all(0.0 <= v <= 1.0 for v in coords):
            continue
        linhas.append("0 " + " ".join(partes[1:5]))

    if not linhas:
        return False

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return True


def pares_em_raiz(raiz: Path):
    labels = {}
    for p in raiz.rglob("*.txt"):
        if p.name.lower() in {"classes.txt", "license.txt"}:
            continue
        labels.setdefault(p.stem, p)

    pares = []
    for img in raiz.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in VALID_EXT:
            continue
        lbl = labels.get(img.stem)
        if lbl:
            pares.append((img, lbl))
    return pares


def detectar_split(path: Path):
    partes = {p.lower() for p in path.parts}
    for destino, aliases in SPLIT_ALIASES.items():
        if partes & aliases:
            return destino
    return None


def copiar_pares(itens, destino: Path, split: str):
    img_dst = destino / "images" / split
    lbl_dst = destino / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    total = 0
    usados = set()
    for img, lbl in itens:
        nome = img.name
        if nome in usados or (img_dst / nome).exists():
            nome = f"{img.parent.name}_{img.name}"
        usados.add(nome)
        stem = Path(nome).stem
        if normalizar_label(lbl, lbl_dst / f"{stem}.txt"):
            shutil.copy2(img, img_dst / nome)
            total += 1
    return total


def preparar(raiz: Path, destino: Path, seed: int = 42):
    pares = pares_em_raiz(raiz)
    if not pares:
        raise RuntimeError(
            "Nenhum par imagem/label YOLO encontrado no dataset baixado. "
            "Verifique a estrutura publicada pelo provedor."
        )

    if destino.exists():
        shutil.rmtree(destino)

    por_split = {"train": [], "val": [], "test": []}
    sem_split = []
    for img, lbl in pares:
        split = detectar_split(img)
        if split:
            por_split[split].append((img, lbl))
        else:
            sem_split.append((img, lbl))

    tem_splits_validos = len(por_split["train"]) > 0 and len(por_split["val"]) > 0

    if not tem_splits_validos:
        todos = pares[:]
        random.Random(seed).shuffle(todos)
        n = len(todos)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        por_split = {
            "train": todos[:n_train],
            "val": todos[n_train:n_train + n_val],
            "test": todos[n_train + n_val:],
        }
    elif sem_split:
        por_split["train"].extend(sem_split)

    totais = {}
    for split in ("train", "val", "test"):
        totais[split] = copiar_pares(por_split[split], destino, split)
        print(f"{split}: {totais[split]} imagens/labels validos")

    if totais["train"] == 0 or totais["val"] == 0:
        raise RuntimeError("Dataset preparado sem train/val validos.")

    if totais["test"] == 0:
        print("Aviso: split test vazio; a validacao final de test sera ignorada.")

    Path("dataset.yaml").write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )

    print(f"Dataset pronto em {destino.resolve()}")
    print("Classe final: 0 = license_plate")


def main():
    parser = argparse.ArgumentParser(
        description="Baixa e prepara dataset publico de deteccao de placas sem Roboflow/API key."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--destino", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raiz = baixar(args.dataset)
    preparar(raiz, Path(args.destino), args.seed)


if __name__ == "__main__":
    main()
