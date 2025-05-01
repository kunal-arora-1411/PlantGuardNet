Here's the updated **README.md** content for your `PlantGuardNet` project with:

- 🌐 Link to the `.pth` model file
- 📷 Screenshot section for UI display
- Polished layout and consistency

---

```markdown
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



## 🧩 Model Checkpoint

You can download the trained PyTorch model (`plant_disease_model.pth`) from the following link:

🔗 [Download Model (.pth)](https://drive.google.com/file/d/1j7aWUiyAGlVr-LY81f3_FZktCd5zUC-S/view?usp=sharing)

> Place the `.pth` file inside your project directory where the model is loaded.

---

## 📸 Screenshots

| Upload Page | Prediction & Recommendation |
|-------------|-----------------------------|
| ![Upload](screenshots/upload.png) | ![Result](screenshots/result.png) |

> Save your UI screenshots as `upload.png` and `result.png` in a folder named `screenshots/`.

---

## 🛠️ Installation

```bash
git clone https://github.com/kunal-arora-1411/PlantGuardNet.git
cd PlantGuardNet
pip install -r requirements.txt
```

---

## 🏃‍♂️ Run the App

```bash
streamlit run app.py
```

---

## 📫 Contact

Feel free to connect or raise issues if you have suggestions or improvements!

```

---

Let me know if you'd like the README in `.md` file format or want help resizing your UI screenshots for GitHub display.
