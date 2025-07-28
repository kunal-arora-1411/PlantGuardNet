# ===== 1. Imports =====
import os
import requests
from tqdm import tqdm

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===== 2. Constants for Downloading Weights =====
MODEL_PATH = "best_model.pth"
RELEASE_URL = (
    "https://github.com/kunal-arora-1411/PlantGuardNet/releases/download/v1.0/best_model.pth"
)

def download_weights(url: str = RELEASE_URL, dest: str = MODEL_PATH):
    """Download the model weights from GitHub Releases with a progress bar."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def get_model_path() -> str:
    """Ensure the model weights are present locally, downloading if needed."""
    if not os.path.exists(MODEL_PATH):
        download_weights()
    return MODEL_PATH

# ===== 3. Define Model Class =====
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ===== 4. Load Model and Settings =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight",       "Potato___Late_blight",
    "Potato___healthy",            "Tomato_Bacterial_spot",
    "Tomato_Early_blight",         "Tomato_Late_blight",
    "Tomato_Leaf_Mold",            "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",          "Tomato_Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Tomato_mosaic_virus",  "Tomato_healthy"
]

# download if needed, then load
model_path = get_model_path()
model = PlantCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ===== 5. Full Plant Info =====
# … (your existing plant_info dict unchanged) …

# ===== 6. Streamlit App Frontend =====
st.set_page_config(page_title="PlantGuardNet 🌿", layout="wide")

with st.sidebar:
    st.title("📘 About PlantGuardNet")
    st.markdown("""
    **PlantGuardNet** is a deep learning-powered disease detection system for tomato, potato, and pepper plants.
    
    ✅ Upload a leaf image  
    🧠 Detect the disease using a CNN  
    🌾 Get care tips on soil, nutrients, fertilizers, and more

    Built using **Streamlit** and **PyTorch**, this app helps promote precision agriculture and crop health.
    """)

st.title("🌿 PlantGuardNet: Plant Disease Classifier & Care Guide")
st.write("Upload a plant leaf image to predict its disease and receive tailored farming tips.")

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="📷 Uploaded Leaf", use_container_width=True)
    with col2:
        with st.spinner('🔍 Predicting...'):
            image = Image.open(uploaded_file).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                _, idx = torch.max(outputs, 1)
                predicted_class = class_names[idx.item()]
            st.success(f"✅ **Predicted Class:** {predicted_class}")

            info = plant_info.get(predicted_class)
            if info:
                st.subheader("🧪 Soil Type");           st.info(info["Soil_Type"])
                st.subheader("🥔 Nutritional Needs");   st.success(info["Nutritional_Needs"])
                st.subheader("🧴 Fertilizer Recommendation"); st.warning(info["Fertilizer"])
                st.subheader("💧 Water Level Needed");  st.write(info.get("Water_Level","—"))
                if "Pesticide" in info:
                    st.subheader("🛡️ Pesticide Recommendation"); st.error(info["Pesticide"])
                with st.expander("✅ Do's"):
                    for do in info["Dos"]: st.markdown(f"- {do}")
                with st.expander("⛔ Don'ts"):
                    for dont in info["Donts"]: st.markdown(f"- {dont}")
            else:
                st.warning("⚠️ No farming info available yet for this class.")
else:
    st.info("👆 Please upload an image to get started!")

st.caption("🧠 Built with ❤️ using Streamlit and PyTorch | © 2025 PlantGuardNet")
