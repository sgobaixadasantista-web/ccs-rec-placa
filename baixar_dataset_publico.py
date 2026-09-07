import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nx9xbs4rgx-2.zip"
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def baixar(url: str, destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        print(f"Download já existe: {destino}")
        return
    print(f"Baixando dataset público: {url}")
    urllib.request.urlretrieve(url, destino)
    print(f"Download concluído: {destino} ({destino.stat().st_size / 1024**2:.1f} MB)")


def extrair(zip_path: Path, destino: Path):
    marcador = destino / ".extraido"
    if marcador.exists():
        print("Dataset já extraído.")
        return
    destino.mkdir(parents=True, exist_ok=True)
    print("Extraindo dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destino)
    marcador.write_text("ok\n", encoding="utf-8")


def localizar_pastas(raiz: Path):
    dirs = [p for p in raiz.rglob("*") if p.is_dir()]
    imagens = [p for p in dirs if p.name.lower() == "images"]
    labels = [p for p in dirs if p.name.lower() == "labels"]

    for img_dir in imagens:
        for lbl_dir in labels:
            if img_dir.parent == lbl_dir.parent:
                return img_dir, lbl_dir

    if imagens and labels:
        return imagens[0], labels[0]
    raise RuntimeError("Não foi possível localizar as pastas Images/Labels no dataset extraído.")


def coletar_pares(images_dir: Path, labels_dir: Path):
    pares = []
    labels_por_stem = {p.stem: p for p in labels_dir.rglob("*.txt")}
    for img in images_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in VALID_EXT:
            lbl = labels_por_stem.get(img.stem)
            if lbl:
                pares.append((img, lbl))
    return pares


def normalizar_label(origem: Path, destino: Path):
    linhas = []
    for linha in origem.read_text(encoding="utf-8", errors="ignore").splitlines():
        partes = linha.strip().split()
        if len(partes) < 5:
            continue
        partes[0] = "0"
        linhas.append(" ".join(partes[:5]))
    destino.write_text(("\n".join(linhas) + "\n") if linhas else "", encoding="utf-8")


def preparar(extract_root: Path, destino: Path, seed: int = 42):
    images_dir, labels_dir = localizar_pastas(extract_root)
    pares = coletar_pares(images_dir, labels_dir)
    if not pares:
        raise RuntimeError("Nenhum par imagem/label encontrado.")

    random.Random(seed).shuffle(pares)
    n = len(pares)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": pares[:n_train],
        "val": pares[n_train:n_train + n_val],
        "test": pares[n_train + n_val:],
    }

    if destino.exists():
        shutil.rmtree(destino)

    for split, itens in splits.items():
        img_dst = destino / "images" / split
        lbl_dst = destino / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img, lbl in itens:
            shutil.copy2(img, img_dst / img.name)
            normalizar_label(lbl, lbl_dst / f"{img.stem}.txt")
        print(f"{split}: {len(itens)} imagens")

    yaml = Path("dataset.yaml")
    yaml.write_text(
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
    parser = argparse.ArgumentParser(description="Baixa e prepara o Artificial Mercosur License Plates sem Roboflow.")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--zip", default="data/raw/artificial-mercosur.zip")
    parser.add_argument("--extract", default="data/raw/artificial-mercosur")
    parser.add_argument("--destino", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    extract_root = Path(args.extract)
    destino = Path(args.destino)

    baixar(args.url, zip_path)
    extrair(zip_path, extract_root)
    preparar(extract_root, destino, args.seed)


if __name__ == "__main__":
    main()
