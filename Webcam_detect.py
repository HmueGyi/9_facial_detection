from facenet_pytorch import MTCNN
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import argparse
import os
import time

# Emotion classes
class_names = [
    'Angry', 'Contempt', 'Disgust', 'Fear',
    'Happy', 'Natural', 'Sad', 'Sleepy', 'Surprised'
]

# Transform
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])


def parse_args():
    parser = argparse.ArgumentParser(description='Webcam / Video emotion detection with CLI')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to model weights (default: best_model.pth)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: camera index (0,1,...) or path to video/image (default: 0)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Device to run on: 'auto' (default), 'cpu' or 'cuda'")
    parser.add_argument('--width', type=int, default=0, help='Set capture width (optional)')
    parser.add_argument('--height', type=int, default=0, help='Set capture height (optional)')
    parser.add_argument('--conf', type=float, default=0.5, help='Min face detection confidence (0-1)')
    parser.add_argument('--save-output', type=str, default='',
                        help='Optional path to save annotated output video (e.g. out.mp4)')
    return parser.parse_args()


def load_model(model_path, device):
    num_classes = len(class_names)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f'Error loading model: {e}')
        return

    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)

    # Prepare source
    src = args.source
    if src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'Unable to open source: {args.source}')
        return

    # Optionally set capture size
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save_output:
        # Prepare video writer (use frame size after reading first frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print('Press q in the window to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) if isinstance(src, int) else frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Detect faces and confidences
        boxes, probs = mtcnn.detect(img)

        if boxes is not None:
            # Filter by confidence if probs available
            for i, box in enumerate(boxes):
                prob = probs[i] if probs is not None else 1.0
                if prob < args.conf:
                    continue

                x1, y1, x2, y2 = map(int, box)
                # Clip coords
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, frame.shape[1])
                y2 = min(y2, frame.shape[0])

                try:
                    face = img.crop((x1, y1, x2, y2))
                    input_tensor = transform(face).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        label = class_names[predicted.item()]
                except Exception:
                    label = 'Error'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} ({prob:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Initialize writer after getting frame size
        if args.save_output and writer is None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            writer = cv2.VideoWriter(args.save_output, fourcc, fps, (w, h))

        if writer is not None:
            writer.write(frame)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()