import argparse
import random
import shutil
from pathlib import Path

EXTENSOES = {'.jpg', '.jpeg', '.png', '.webp'}


def preparar(origem: Path, destino: Path, val_ratio: float = 0.2, seed: int = 42):
    imagens = [p for p in origem.iterdir() if p.suffix.lower() in EXTENSOES]
    if not imagens:
        raise SystemExit('Nenhuma imagem encontrada na pasta de origem.')

    random.Random(seed).shuffle(imagens)
    qtd_val = max(1, round(len(imagens) * val_ratio)) if len(imagens) > 1 else 0
    val = set(imagens[:qtd_val])

    for split in ('train', 'val'):
        (destino / 'images' / split).mkdir(parents=True, exist_ok=True)
        (destino / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for imagem in imagens:
        split = 'val' if imagem in val else 'train'
        shutil.copy2(imagem, destino / 'images' / split / imagem.name)

        label = imagem.with_suffix('.txt')
        if label.exists():
            shutil.copy2(label, destino / 'labels' / split / label.name)
        else:
            print(f'AVISO: sem anotacao YOLO para {imagem.name}')

    print(f'Total: {len(imagens)} | treino: {len(imagens)-len(val)} | validacao: {len(val)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organiza imagens e labels em train/val para YOLO')
    parser.add_argument('origem', help='Pasta contendo imagens e arquivos .txt YOLO com mesmo nome')
    parser.add_argument('--destino', default='dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    preparar(Path(args.origem), Path(args.destino), args.val_ratio, args.seed)
