from ultralytics import YOLO


def main():
    # Modelo base pequeno para iniciar o fine-tuning.
    # O treinamento gera best.pt em runs/detect/...
    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        project="runs",
        name="placas",
    )


if __name__ == "__main__":
    main()
