# ===== 1. Imports =====
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===== 2. Define Model Class =====
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

# ===== 3. Load Model and Settings =====
device = "cuda" if torch.cuda.is_available() else "cpu"

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

model = PlantCNN(num_classes=len(class_names))
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===== 4. Full Plant Info =====
plant_info = { 
    "Pepper__bell___Bacterial_spot": {
        "Soil_Type": "Well-drained loamy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Balanced N-P-K like 10-10-10",
        "Fertilizer": "Compost + ammonium nitrate side-dress",
        "Water_Level": "Moderate; consistent soil moisture",
        "Pesticide": "Copper-based bactericide spray weekly",
        "Dos": [
            "Use certified disease-free seeds",
            "Apply drip irrigation",
            "Maintain proper nitrogen levels"
        ],
        "Donts": [
            "Avoid overhead watering",
            "Do not handle plants when wet"
        ]
    },
    "Pepper__bell___healthy": {
        "Soil_Type": "Well-drained loamy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Maintain balanced NPK and micronutrients",
        "Fertilizer": "Organic compost or 10-10-10 fertilizer",
        "Water_Level": "Moderate; regular watering without wetting leaves",
        "Dos": [
            "Water consistently without wetting leaves",
            "Mulch to retain soil moisture"
        ],
        "Donts": [
            "Avoid water stress",
            "Avoid wetting foliage directly"
        ]
    },
    "Potato___Early_blight": {
        "Soil_Type": "Sandy loam soil (pH 5.0–6.0)",
        "Nutritional_Needs": "High phosphorus and potassium, moderate nitrogen",
        "Fertilizer": "Apply 5-10-10 fertilizer at planting",
        "Water_Level": "Moderate; avoid excess moisture",
        "Pesticide": "Chlorothalonil fungicide at early signs",
        "Dos": [
            "Rotate crops yearly",
            "Use certified disease-free seed potatoes"
        ],
        "Donts": [
            "Avoid excess nitrogen fertilization",
            "Do not leave infected plant debris"
        ]
    },
    "Potato___Late_blight": {
        "Soil_Type": "Sandy loam soil (pH 5.0–6.0)",
        "Nutritional_Needs": "Emphasize potassium supplementation",
        "Fertilizer": "Apply potassium-rich fertilizers like muriate of potash",
        "Water_Level": "Low to moderate; avoid wet foliage",
        "Pesticide": "Mancozeb or copper fungicide sprays",
        "Dos": [
            "Spray fungicides at early signs",
            "Destroy infected plants immediately"
        ],
        "Donts": [
            "Avoid planting potatoes too densely",
            "Don't overhead irrigate during wet weather"
        ]
    },
    "Potato___healthy": {
        "Soil_Type": "Sandy loam, pH 5.0–6.0",
        "Nutritional_Needs": "Balanced 5-10-10 NPK supply with organic matter",
        "Fertilizer": "Compost + balanced fertilizer before planting",
        "Water_Level": "Moderate; ensure well-drained soil",
        "Dos": [
            "Maintain loose, well-aerated soil"
        ],
        "Donts": [
            "Avoid overwatering and waterlogging"
        ]
    },
    "Tomato_Bacterial_spot": {
        "Soil_Type": "Well-drained fertile soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Balanced NPK with calcium",
        "Fertilizer": "Apply 10-10-10 pre-planting, side-dress calcium nitrate",
        "Water_Level": "Moderate; drip irrigation preferred",
        "Pesticide": "Copper-based bactericide weekly",
        "Dos": [
            "Use copper-based bactericides",
            "Grow resistant varieties if available"
        ],
        "Donts": [
            "Avoid overhead irrigation",
            "Don't compost infected plant material"
        ]
    },
    "Tomato_Early_blight": {
        "Soil_Type": "Loamy soil rich in organic matter (pH 6.2–6.8)",
        "Nutritional_Needs": "Slightly higher phosphorus and potassium",
        "Fertilizer": "Phosphorus-rich fertilizers like 10-20-10",
        "Water_Level": "Moderate; maintain soil moisture without wetting leaves",
        "Pesticide": "Chlorothalonil or copper fungicide sprays",
        "Dos": [
            "Mulch heavily to prevent soil splash",
            "Remove lower leaves showing symptoms"
        ],
        "Donts": [
            "Do not reuse stakes or cages without sterilization",
            "Avoid watering late in the day"
        ]
    },
    "Tomato_Late_blight": {
        "Soil_Type": "Loamy soil with excellent drainage (pH 6.2–6.8)",
        "Nutritional_Needs": "Potassium-heavy diet, moderate nitrogen",
        "Fertilizer": "Use 5-10-10 or potassium sulfate fertilizers",
        "Water_Level": "Moderate; dry foliage quickly after rain",
        "Pesticide": "Mancozeb or metalaxyl fungicide",
        "Dos": [
            "Apply protective fungicide sprays",
            "Remove infected leaves immediately"
        ],
        "Donts": [
            "Avoid dense planting",
            "Don't allow leaves to stay wet overnight"
        ]
    },
    "Tomato_Leaf_Mold": {
        "Soil_Type": "Well-drained soil with good air circulation (pH 6.0–6.8)",
        "Nutritional_Needs": "Moderate NPK balance with extra calcium",
        "Fertilizer": "Balanced 10-10-10 fertilizer + calcium source",
        "Water_Level": "Moderate; prevent humidity buildup",
        "Pesticide": "Fungicides like chlorothalonil or mancozeb",
        "Dos": [
            "Ensure good air movement by pruning",
            "Use resistant tomato varieties"
        ],
        "Donts": [
            "Avoid watering late evening",
            "Don't grow too close together"
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "Soil_Type": "Loamy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Moderate nitrogen, high phosphorus",
        "Fertilizer": "10-20-10 type fertilizers at planting",
        "Water_Level": "Moderate; avoid water splashing",
        "Pesticide": "Fungicides like chlorothalonil or copper",
        "Dos": [
            "Rotate crops every year",
            "Remove infected leaves quickly"
        ],
        "Donts": [
            "Avoid handling wet plants",
            "Don't let weeds grow near tomatoes"
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Soil_Type": "Loose, well-drained soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Nitrogen-rich early growth, balanced later",
        "Fertilizer": "Initial 10-10-10 followed by side-dressing nitrogen",
        "Water_Level": "Moderate; avoid drought stress",
        "Pesticide": "Miticides (like abamectin) if infestation severe",
        "Dos": [
            "Spray water jets to dislodge mites",
            "Introduce beneficial insects like ladybugs"
        ],
        "Donts": [
            "Avoid pesticide overuse that kills beneficial insects",
            "Don't allow water stress (mites thrive on dry plants)"
        ]
    },
    "Tomato_Target_Spot": {
        "Soil_Type": "Loamy and well-drained soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Maintain strong phosphorus and potassium levels",
        "Fertilizer": "10-20-20 fertilizers preferred",
        "Water_Level": "Moderate; avoid late evening watering",
        "Pesticide": "Protective fungicides like mancozeb",
        "Dos": [
            "Practice crop rotation",
            "Apply preventive fungicides"
        ],
        "Donts": [
            "Avoid late irrigation",
            "Don't overcrowd plants"
        ]
    },
    "Tomato_Tomato_YellowLeaf__Curl_Virus": {
        "Soil_Type": "Light, well-drained sandy soil (pH 6.0–6.8)",
        "Nutritional_Needs": "Balanced NPK, maintain calcium and magnesium",
        "Fertilizer": "Use 10-10-10 fertilizers with micronutrients",
        "Water_Level": "Moderate; ensure even moisture",
        "Pesticide": "Insecticides (like imidacloprid) for whiteflies",
        "Dos": [
            "Use insect-proof netting against whiteflies",
            "Remove infected plants early"
        ],
        "Donts": [
            "Do not plant near older infected crops",
            "Don't ignore whitefly infestations"
        ]
    },
    "Tomato_Tomato_mosaic_virus": {
        "Soil_Type": "Fertile, well-drained loamy soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Balanced nutrients with micronutrients support",
        "Fertilizer": "10-10-10 fertilizer with added micronutrients",
        "Water_Level": "Moderate; regular watering without splashing leaves",
        "Pesticide": "No effective pesticide; manage by hygiene",
        "Dos": [
            "Disinfect tools regularly",
            "Grow resistant varieties if available"
        ],
        "Donts": [
            "Don't smoke near plants (virus transmission risk)",
            "Avoid handling healthy plants after infected ones"
        ]
    },
    "Tomato_healthy": {
        "Soil_Type": "Loamy, nutrient-rich soil (pH 6.2–6.8)",
        "Nutritional_Needs": "Balanced NPK with micronutrient support",
        "Fertilizer": "Base fertilizer 10-10-10 + monthly side-dressing calcium nitrate",
        "Water_Level": "Moderate; maintain consistent soil moisture",
        "Dos": [
            "Stake or cage plants for support",
            "Mulch around base to conserve moisture"
        ],
        "Donts": [
            "Avoid waterlogging",
            "Don't let leaves touch the ground"
        ]
    }
}

# ===== 5. Streamlit App Frontend =====
import streamlit as st
from PIL import Image
import torch

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="PlantGuardNet 🌿", layout="wide")

# Sidebar: About the project
with st.sidebar:
    st.title("📘 About PlantGuardNet")
    st.markdown("""
    **PlantGuardNet** is a deep learning-powered disease detection system for tomato, potato, and pepper plants.
    
    ✅ Upload a leaf image  
    🧠 Detect the disease using a CNN  
    🌾 Get care tips on soil, nutrients, fertilizers, and more

    Built using **Streamlit** and **PyTorch**, this app helps promote precision agriculture and crop health.
    """)

# Main UI
st.title("🌿 PlantGuardNet: Plant Disease Classifier & Care Guide")
st.write("Upload a plant leaf image to predict its disease and receive tailored farming tips.")

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="📷 Uploaded Leaf", use_container_width=True)  # ✅ fixed

    with col2:
        with st.spinner('🔍 Predicting...'):
            image = Image.open(uploaded_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_names[predicted.item()]

            st.success(f"✅ **Predicted Class:** {predicted_class}")

            info = plant_info.get(predicted_class)
            if info:
                st.subheader("🧪 Soil Type")
                st.info(info["Soil_Type"])

                st.subheader("🥔 Nutritional Needs")
                st.success(info["Nutritional_Needs"])

                st.subheader("🧴 Fertilizer Recommendation")
                st.warning(info["Fertilizer"])

                st.subheader("💧 Water Level Needed")
                st.write(info.get("Water_Level", "Not Available"))

                if "Pesticide" in info:
                    st.subheader("🛡️ Pesticide Recommendation")
                    st.error(info["Pesticide"])

                with st.expander("✅ Do's"):
                    for do in info["Dos"]:
                        st.markdown(f"- {do}")

                with st.expander("⛔ Don'ts"):
                    for dont in info["Donts"]:
                        st.markdown(f"- {dont}")
            else:
                st.warning("⚠️ No farming info available yet for this class.")
else:
    st.info("👆 Please upload an image to get started!")

st.caption("🧠 Built with ❤️ using Streamlit and PyTorch | © 2025 PlantGuardNet")
