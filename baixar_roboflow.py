import argparse
import os
import shutil
from pathlib import Path

from roboflow import Roboflow

DEFAULT_WORKSPACE = "akler"
DEFAULT_PROJECT = "lpr-placas-brasileiras-2025-y4ydq"
DEFAULT_VERSION = 8
DEFAULT_FORMAT = "yolov11"


def normalizar_labels_para_uma_classe(destino: Path) -> int:
    """Converte qualquer classe YOLO do dataset para classe 0 = license_plate."""
    alterados = 0
    for txt in destino.rglob("*.txt"):
        linhas_novas = []
        mudou = False
        for linha in txt.read_text(encoding="utf-8").splitlines():
            partes = linha.strip().split()
            if len(partes) < 5:
                continue
            if partes[0] != "0":
                mudou = True
            partes[0] = "0"
            linhas_novas.append(" ".join(partes))
        if linhas_novas:
            txt.write_text("\n".join(linhas_novas) + "\n", encoding="utf-8")
            if mudou:
                alterados += 1
    return alterados


def localizar_raiz_dataset(base: Path) -> Path:
    candidatos = [base]
    candidatos.extend([p for p in base.iterdir() if p.is_dir()])
    for p in candidatos:
        if (p / "train").exists() or (p / "valid").exists() or (p / "test").exists():
            return p
    return base


def copiar_split(origem: Path, destino: Path, split_origem: str, split_destino: str):
    img_src = origem / split_origem / "images"
    lbl_src = origem / split_origem / "labels"
    if not img_src.exists() or not lbl_src.exists():
        return 0

    img_dst = destino / "images" / split_destino
    lbl_dst = destino / "labels" / split_destino
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for arq in img_src.iterdir():
        if arq.is_file():
            shutil.copy2(arq, img_dst / arq.name)
            total += 1
    for arq in lbl_src.glob("*.txt"):
        shutil.copy2(arq, lbl_dst / arq.name)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Baixa dataset brasileiro de placas do Roboflow e converte para o layout do projeto."
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--download-dir", default="dataset/roboflow_download")
    parser.add_argument("--destino", default="dataset")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="API key do Roboflow. Preferir variável de ambiente ROBOFLOW_API_KEY.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY não configurada. Defina a variável de ambiente ou use --api-key."
        )

    download_dir = Path(args.download_dir)
    destino = Path(args.destino)
    download_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    projeto = rf.workspace(args.workspace).project(args.project)
    versao = projeto.version(args.version)
    dataset = versao.download(args.format, location=str(download_dir))

    raiz = localizar_raiz_dataset(Path(dataset.location))

    total_train = copiar_split(raiz, destino, "train", "train")
    total_val = copiar_split(raiz, destino, "valid", "val")
    total_test = copiar_split(raiz, destino, "test", "test")

    alterados = normalizar_labels_para_uma_classe(destino)

    print("Dataset preparado.")
    print(f"train: {total_train} imagens")
    print(f"val:   {total_val} imagens")
    print(f"test:  {total_test} imagens")
    print(f"labels remapeados para classe única: {alterados}")
    print("Classe final: 0 = license_plate")


if __name__ == "__main__":
    main()
