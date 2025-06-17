import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
# import numpy as np

# Define emotion class names
class_names = [
    'Angry',
    'Contempt',
    'Disgust',
    'Fear',
    'Happy',
    'Natural',
    'Sad',
    'Sleepy',
    'Surprised'
]

# Define image size and transform
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load the trained model
num_classes = len(class_names) 
model = models.resnet18(weights=False)
num_ftrs = model.fc.in_features         # Final layer ထဲသွင်းရမယ့် input size ကို သိဖို့။
model.fc = nn.Linear(num_ftrs, num_classes)   # Final fully-connected layer ကို ၉ emotion classes နဲ့ ကိုက်အောင် ပြောင်း။
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval().to(device)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:                     # ret = return value
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))           # BGR frame ကို RGB ပြောင်းပြီး PIL image ပြုလုပ်တယ်။

    # transform(img)	[3, 224, 224]	1 image
    # .unsqueeze(0)	[1, 3, 224, 224]	batch of 1 image
    
    input_tensor = transform(img).unsqueeze(0).to(device)



    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        label = class_names[predicted_class.item()]

    # Show predicted label on video
    cv2.putText(frame, f'Emotion: {label}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
