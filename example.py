import argparse

from ultralytics import YOLO


def main(epochs: int, imgsz: int, device: str):
    """
    Train a YOLO model with the given epochs and image size.

    Args:
        epochs (int): Number of epochs for training.
        imgsz (int): Image size for training.
        device (str): Device for training.
    """
    # Load a model
    model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='coco128.yaml', epochs=epochs, imgsz=imgsz, device=device)
    print(f"Training complete. Results: {results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a YOLO model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--device', type=str, default='', help='Device for training.')
    args = parser.parse_args()

    main(args.epochs, args.imgsz, args.device)