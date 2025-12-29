import cv2
import torch
import numpy as np
from models import StudentModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SECTORS = 12

def preprocess_slitlamp(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    return img.to(DEVICE)

def load_model():
    model = StudentModel(NUM_SECTORS)
    model.load_state_dict(torch.load(r"C:\Users\shivam.prajapati\Documents\lvp-projects\LUPI_Sutures\student_angular_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(path):
    model = load_model()
    x = preprocess_slitlamp(path)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    return pred

if __name__ == "__main__":
    slit_img = r"C:\Users\shivam.prajapati\Desktop\PK_SL_Pentacam\P1785015\2320870_0.714548001762423152.jpg"
    angular_scores = predict(slit_img)

    print("\nPredicted high-tension regions:")
    for i, v in enumerate(angular_scores):
        print(f"Sector {i+1} ({i}-{i+1} o'clock): {v:.2f}")
