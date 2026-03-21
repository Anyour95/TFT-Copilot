import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.yaml', help='path to dataset yaml')
    p.add_argument('--model', default='yolov8n.pt', help='base model or pretrained weights')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='exp')
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)

if __name__ == '__main__':
    main()
