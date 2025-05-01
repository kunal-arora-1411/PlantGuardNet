# 🌿 PlantGuardNet

**PlantGuardNet** is a deep learning-based system for intelligent plant disease detection and care recommendation. It uses Convolutional Neural Networks (CNN) and segmentation models (SAM) to classify crop types and detect diseases from leaf images, providing tailored agronomic advice.

---

## 🧠 Key Features

- 📸 Leaf-based disease classification using CNN
- 🧪 Integration with crop-specific knowledge base (soil, nutrients, pesticides)
- 🤖 SAM (Segment Anything Model) for precise leaf patch extraction
- 🔁 Dynamic recommendation system (soil type, water, fertilizer, pesticide)
- 💻 Streamlit-based user interface

---

## 🚀 Workflow

1. **Data Collection**  
   - PlantVillage dataset for leaf images  
   - Curated agricultural data for nutrient/pesticide mapping  

2. **Preprocessing**  
   - Image resizing, normalization, augmentation  
   - Cleaning and structuring agri-data  

3. **Model Training**  
   - CNN for crop and disease classification  
   - SAM for segmenting leaf patches (if enabled)

4. **Inference Pipeline**  
   - User uploads image  
   - Model detects crop & disease  
   - System suggests appropriate actions (Do’s, Don’ts, pesticide, fertilizer)

5. **Deployment**  
   - Frontend: Streamlit  
   - Backend: PyTorch model + data engine  

---

## 🛠️ Installation

```bash
git clone https://github.com/kunal-arora-1411/PlantGuardNet.git
cd PlantGuardNet
pip install -r requirements.txt
