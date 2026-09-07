import argparse
from pathlib import Path

from ultralytics import YOLO


MODELO_BASE_PADRAO = "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do detector de placas")
    parser.add_argument("--data", default="dataset.yaml")
    parser.add_argument("--modelo", default=MODELO_BASE_PADRAO)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="placas")
    args = parser.parse_args()

    modelo = YOLO(args.modelo)
    resultados = modelo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=12,
        cache=False,
        workers=2,
        plots=True,
    )

    pasta = Path(args.project) / args.name
    best = pasta / "weights" / "best.pt"
    last = pasta / "weights" / "last.pt"

    print("Treinamento concluido.")
    print(f"best.pt: {best}")
    print(f"last.pt: {last}")
    print(f"results: {pasta}")
    print(resultados)


if __name__ == "__main__":
    main()
