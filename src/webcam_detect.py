import argparse
import cv2
import torch
import mediapipe as mp
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.emotion_model import get_model

# Configuration & labels.
# Keep exact dataset folder spelling/order from training (`ImageFolder.classes`).
# NOTE: `suprised` is intentionally preserved to match the training dataset name.
CLASS_NAMES = [
    'angry', 'happy', 'neutral', 'sad', 'suprised', 'tired',
]

# Inference configuration
IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resolve_mediapipe_face_detection():
    # Standard API path for most mediapipe releases.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        return mp.solutions.face_detection

    # Fallback path used in some builds/layouts.
    try:
        from mediapipe.python.solutions import face_detection as mp_face_detection
        return mp_face_detection
    except Exception:
        pass

    mp_file = getattr(mp, "__file__", "unknown")
    mp_version = getattr(mp, "__version__", "unknown")
    raise RuntimeError(
        "MediaPipe face detection API is unavailable in this install "
        f"(module={mp_file}, version={mp_version}). "
        "Reinstall with: pip uninstall -y mediapipe mediapipe-nightly && "
        "pip install mediapipe==0.10.14"
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Webcam/Video Facial Emotion Detection (MediaPipe + EfficientNet)')
    parser.add_argument('--model-path', type=str, default='weights/best_emotion_model.pth',
                        help='Path to model weights (default: weights/best_emotion_model.pth)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: camera index (0,1,...) or file path (default: 0)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help="Device to run classification on (default: auto)")
    parser.add_argument('--conf', type=float, default=0.5, help='MediaPipe face detection confidence (0-1)')
    parser.add_argument('--save-to', type=str, default='',
                        help='Save annotated output to path (e.g. results.mp4)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Path handling
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file NOT found at {model_path}. Please check your path.")
        return

    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load classification model (EfficientNet-V2-S)
    try:
        model = get_model(model_path=str(model_path), device=device, num_classes=len(CLASS_NAMES))
        print(f"EfficientNet-V2-S model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        mp_face_detection = resolve_mediapipe_face_detection()
        detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=args.conf)
    except Exception as e:
        print(f"Error initializing MediaPipe face detector: {e}")
        return

    # Prepare video source
    src = args.source
    if src.isdigit():
        src = int(src)
    
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Failed to open video source: {args.source}")
        return

    writer = None
    print("-" * 30)
    print("Starting Detection (MediaPipe Backend)...")
    print("Press 'q' to quit.")
    print("-" * 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for webcam
        if isinstance(src, int):
            frame = cv2.flip(frame, 1)

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detect faces using MediaPipe
        results = detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                x2, y2 = x1 + width, y1 + height

                # Boundary safety
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if width <= 0 or height <= 0:
                    continue

                try:
                    # 2. Extract and Preprocess Face
                    face_img = rgb_frame[y1:y2, x1:x2]
                    pil_img = Image.fromarray(face_img)
                    input_tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

                    # 3. Predict Emotion
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        label = CLASS_NAMES[predicted.item()]
                        score = torch.softmax(output, dim=1)[0][predicted.item()].item()
                except Exception as e:
                    print(f"Inference error: {e}")
                    label, score = "Error", 0.0

                # 4. Draw results
                score_pct = score * 100
                display_text = f"{label} ({score_pct:.1f}%)"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Output video writing
        if args.save_to and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            writer = cv2.VideoWriter(args.save_to, fourcc, fps, (w, h))

        if writer:
            writer.write(frame)

        cv2.imshow('Facial Emotion Detection (MediaPipe)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    detector.close()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == '__main__':
    main()
