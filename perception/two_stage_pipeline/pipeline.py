import argparse
import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ensure local package imports work when script run from workspace root
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from detector import Detector
from classifier import Classifier
from utils import crop_box, ensure_dir

try:
    import cv2
    from mss import mss
except Exception:
    cv2 = None
    mss = None


def draw_box_label(pil_img, box, label, color=(0, 255, 0), width=2):
    draw = ImageDraw.Draw(pil_img)
    x1, y1, x2, y2 = map(int, box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw.text((x1, y1 - 12), label, fill=color)


def load_names(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='image file or folder')
    p.add_argument('--detector_weights', default='yolov8n.pt')
    p.add_argument('--classifier_weights', default=None)
    p.add_argument('--names', default='names.txt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--output', default='out')
    p.add_argument('--realtime', action='store_true', help='use screen capture realtime mode')
    p.add_argument('--fps', type=float, default=5.0, help='target FPS in realtime mode')
    p.add_argument('--monitor', type=int, default=1, help='mss monitor id (default 1 = full screen)')
    args = p.parse_args()

    detector = Detector(weights=args.detector_weights)
    names = load_names(args.names)
    # num_classes from names if available
    num_classes = len(names) if names else None
    classifier = Classifier(weights_path=args.classifier_weights, num_classes=num_classes)

    ensure_dir(args.output)

    if args.realtime:
        if mss is None or cv2 is None:
            print('Realtime mode requires `mss` and `opencv-python`. Install requirements and retry.')
            return

        interval = 1.0 / max(0.1, args.fps)
        sct = mss()
        monitor = sct.monitors[args.monitor] if args.monitor < len(sct.monitors) else sct.monitors[1]
        window_name = 'TFT Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                t0 = time.time()
                sct_img = sct.grab(monitor)
                # mss image to PIL RGB
                img_pil = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
                img_np = np.array(img_pil)

                try:
                    results = detector.predict(img_np, imgsz=args.imgsz, conf=args.conf)
                    # Use results for the single frame
                    for res in results:
                        img_frame = Image.fromarray(res.orig_img) if getattr(res, 'orig_img', None) is not None else img_pil
                        if hasattr(res, 'boxes') and res.boxes is not None:
                            xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
                            for box in xyxy:
                                cropped = crop_box(img_frame, box)
                                preds = classifier.predict_image(cropped, topk=1, names=names)
                                label, prob = preds[0]
                                draw_box_label(img_frame, box, f'{label} {prob:.2f}')
                        # convert PIL to BGR for OpenCV display
                        disp = cv2.cvtColor(np.array(img_frame), cv2.COLOR_RGB2BGR)
                        cv2.imshow(window_name, disp)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt()
                except Exception as e:
                    print('Frame processing error:', e)

                elapsed = time.time() - t0
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except KeyboardInterrupt:
            print('Realtime inference stopped by user')
        finally:
            cv2.destroyAllWindows()

    else:
        src = Path(args.source)
        paths = [src] if src.is_file() else sorted(src.glob('*'))

        for path in paths:
            try:
                results = detector.predict(str(path), imgsz=args.imgsz, conf=args.conf)
                # ultralytics returns list-like results (one per image)
                for res in results:
                    img_np = res.orig_img  # numpy HWC BGR or RGB depending on version
                    # Convert to PIL RGB
                    if img_np is None:
                        continue
                    img = Image.fromarray(img_np)
                    boxes = []
                    if hasattr(res, 'boxes') and res.boxes is not None:
                        xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
                        for box in xyxy:
                            cropped = crop_box(img, box)
                            preds = classifier.predict_image(cropped, topk=1, names=names)
                            label, prob = preds[0]
                            draw_box_label(img, box, f'{label} {prob:.2f}')
                    out_path = os.path.join(args.output, Path(path).name)
                    img.save(out_path)
            except Exception as e:
                print('Error processing', path, e)


if __name__ == '__main__':
    main()
