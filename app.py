# ===== 1. Imports =====
import os
import requests
from tqdm import tqdm

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===== 2. Downloading Weights from GitHub Release =====
MODEL_PATH = "best_model.pth"
RELEASE_URL = (
    "https://github.com/kunal-arora-1411/PlantGuardNet/"
    "releases/download/v1.0/best_model.pth"
)

def download_weights(url=RELEASE_URL, dest=MODEL_PATH):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def get_model_path():
    if not os.path.exists(MODEL_PATH):
        download_weights()
    return MODEL_PATH

# ===== 3. Define CNN Architecture =====
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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
            nn.Linear(256*14*14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ===== 4. Load Model & Settings =====
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

model = PlantCNN(len(class_names))
model.load_state_dict(torch.load(get_model_path(), map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

# ===== 5. Plant Info Dictionary =====
plant_info = {
    "Pepper__bell___Bacterial_spot": {
        "Soil_Type": "Well-drained loamy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Balanced N-P-K like 10-10-10",
        "Fertilizer": "Compost + ammonium nitrate side-dress",
        "Water_Level": "Moderate; consistent soil moisture",
        "Pesticide": "Copper-based bactericide spray weekly",
        "Dos": ["Use certified disease-free seeds", "Apply drip irrigation", "Maintain proper nitrogen levels"],
        "Donts": ["Avoid overhead watering", "Do not handle plants when wet"]
    },
    "Pepper__bell___healthy": {
        "Soil_Type": "Well-drained loamy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Maintain balanced NPK and micronutrients",
        "Fertilizer": "Organic compost or 10-10-10 fertilizer",
        "Water_Level": "Moderate; regular watering without wetting leaves",
        "Dos": ["Water consistently without wetting leaves", "Mulch to retain soil moisture"],
        "Donts": ["Avoid water stress", "Avoid wetting foliage directly"]
    },
    # … include **all** your other classes here exactly as before …
    "Tomato_healthy": {
        "Soil_Type": "Loamy, nutrient-rich soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Balanced NPK with micronutrient support",
        "Fertilizer": "Base fertilizer 10-10-10 + monthly side-dressing calcium nitrate",
        "Water_Level": "Moderate; maintain consistent soil moisture",
        "Dos": ["Stake or cage plants for support", "Mulch around base to conserve moisture"],
        "Donts": ["Avoid waterlogging", "Don't let leaves touch the ground"]
    }
}

# ===== 6. Streamlit UI =====
st.set_page_config(page_title="PlantGuardNet 🌿", layout="wide")

with st.sidebar:
    st.title("📘 About PlantGuardNet")
    st.markdown("""
    **PlantGuardNet** is a deep learning-powered disease detection system for tomato, potato, and pepper plants.
    Built with **Streamlit** and **PyTorch**.
    """)

st.title("🌿 PlantGuardNet: Plant Disease Classifier & Care Guide")
st.write("Upload a plant leaf image to predict its disease and receive tailored farming tips.")

uploaded = st.file_uploader("📤 Upload a Leaf Image", type=["jpg","jpeg","png"])
if uploaded:
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(uploaded, caption="📷 Uploaded Leaf", use_column_width=True)
    with col2:
        with st.spinner("🔍 Predicting..."):
            img = Image.open(uploaded).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                idx = out.argmax(dim=1).item()
                pred = class_names[idx]
            st.success(f"✅ **{pred}**")

            info = plant_info.get(pred)
            if info:
                st.subheader("🧪 Soil Type");           st.info(info["Soil_Type"])
                st.subheader("🥔 Nutritional Needs");   st.success(info["Nutritional_Needs"])
                st.subheader("🧴 Fertilizer");          st.warning(info["Fertilizer"])
                st.subheader("💧 Water Level");         st.write(info.get("Water_Level","—"))
                if "Pesticide" in info:
                    st.subheader("🛡️ Pesticide");      st.error(info["Pesticide"])
                with st.expander("✅ Do's"):
                    for d in info["Dos"]: st.markdown(f"- {d}")
                with st.expander("⛔ Don'ts"):
                    for d in info["Donts"]: st.markdown(f"- {d}")
            else:
                st.warning("⚠️ No agronomic info available.")
else:
    st.info("👆 Please upload an image to get started!")

st.caption("🧠 Built with ❤️ using Streamlit and PyTorch | © 2025 PlantGuardNet")
